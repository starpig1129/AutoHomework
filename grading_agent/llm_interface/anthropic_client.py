# -*- coding: utf-8 -*-
"""Implementation of the LLMInterface for Anthropic models."""

import logging
from typing import Any, Dict, List

import anthropic

from .base import LLMInput, LLMInterface, LLMOutput

logger = logging.getLogger(__name__)


class AnthropicClient(LLMInterface):
  """Concrete implementation of LLMInterface for Anthropic models using async API."""

  def __init__(self, api_key: str, model_name: str, **kwargs: Any):
    """Initializes the asynchronous Anthropic client.

    Args:
        api_key: The Anthropic API key.
        model_name: The specific Anthropic model to use (e.g., "claude-3-5-sonnet-latest").
        **kwargs: Additional keyword arguments for the Anthropic client.
                  Common arguments include 'max_tokens'.
    """
    super().__init__(api_key=api_key, model_name=model_name, **kwargs)
    try:
      # Note: Anthropic client uses env var ANTHROPIC_API_KEY by default,
      # but we pass it explicitly for clarity and consistency.
      self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
      self.max_tokens = kwargs.get("max_tokens", 1024) # Default from main.py
      logger.info(
          "AsyncAnthropic client initialized successfully for model %s.",
          self.model_name,
      )
    except Exception as e:
      logger.exception(
          "Failed to initialize AsyncAnthropic client for model %s: %s",
          self.model_name,
          e,
      )
      raise

  async def _prepare_request_payload(
      self, system_prompt: str, requirements: str, llm_input: LLMInput
  ) -> Dict[str, Any]:
    """Prepares the request payload for the Anthropic Messages API.

    Args:
        system_prompt: The system prompt (used in the system parameter).
        requirements: The assignment requirements.
        llm_input: The standardized input data.

    Returns:
        A dictionary representing the request payload.
    """
    # Anthropic's Messages API uses a 'system' parameter and a list of 'messages'.
    # We'll construct the user message content combining text and images.

    user_message_content: List[Dict[str, Any]] = []

    # Add text parts first, combining requirements and student info
    text_block = f"作業要求：\n{requirements}\n\n"
    text_block += f"學生資訊：{llm_input.student_id} - {llm_input.student_name}\n\n"

    if llm_input.text_content:
        text_block += "作業文字內容：\n" + "\n".join(llm_input.text_content)
    else:
        text_block += "無文字內容提交。"

    # Add the main text block
    user_message_content.append({"type": "text", "text": text_block})


    # Add image parts
    if llm_input.image_content:
        user_message_content.append({"type": "text", "text": "作業圖片內容如下："})
        for img_base64 in llm_input.image_content:
            # Ensure base64 string doesn't contain data URI prefix
            if "base64," in img_base64:
                img_base64 = img_base64.split("base64,")[-1]
            user_message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    # TODO: Determine media type more accurately if possible
                    "media_type": "image/jpeg",
                    "data": img_base64,
                },
            })
    else:
        user_message_content.append({"type": "text", "text": "無圖片內容提交。"})


    # Final instruction part
    user_message_content.append({"type": "text", "text": "請評分這位學生的作業。"})


    payload = {
        "model": self.model_name,
        "system": system_prompt, # Use the dedicated system prompt parameter
        "messages": [{"role": "user", "content": user_message_content}],
        "max_tokens": self.max_tokens,
        # Anthropic doesn't typically use 'temperature' in the same way,
        # but you could add other relevant parameters here if needed.
    }
    logger.debug(
        "Prepared Anthropic request payload for student %s.", llm_input.student_id
    )
    # logger.debug("Payload messages: %s", user_message_content) # Careful with image data
    return payload

  async def _parse_response(
      self, response: anthropic.types.Message, student_id: str
  ) -> LLMOutput:
    """Parses the Anthropic API response.

    Args:
        response: The Message object from the Anthropic API.
        student_id: The student ID.

    Returns:
        An LLMOutput object.
    """
    try:
      # Anthropic response structure: response.content is a list of blocks
      if not response.content or not isinstance(response.content[0], anthropic.types.TextBlock):
          logger.warning("Received unexpected response format or empty content block for student %s.", student_id)
          return LLMOutput(student_id=student_id, error="Invalid or empty response block from Anthropic")

      raw_text = response.content[0].text
      if not raw_text:
        logger.warning(
            "Received empty text in response block for student %s.", student_id
        )
        return LLMOutput(
            student_id=student_id, error="Empty response text from Anthropic"
        )

      # Clean the response
      cleaned_text = raw_text.strip().strip('"`').strip()
      logger.info(
          "Raw response for student %s: %s", student_id, cleaned_text
      )

      parts = [p.strip() for p in cleaned_text.split(',')]
      if len(parts) >= 3:
        parsed_student_id = parts[0]
        # Optional: Validate student ID
        # if parsed_student_id != student_id:
        #     logger.warning("Mismatched student ID in Anthropic response for %s: got %s", student_id, parsed_student_id)

        try:
          score = int(parts[1])
          if not 0 <= score <= 100:
             logger.warning("Score out of range for student %s: %d", student_id, score)
             score = max(0, min(100, score)) # Clamp score
        except ValueError:
          logger.error(
              "Failed to parse score as integer for student %s: %s",
              student_id,
              parts[1],
          )
          return LLMOutput(
              student_id=student_id,
              error=f"Invalid score format: {parts[1]}",
              raw_response=raw_text,
          )

        comment = parts[2]
        # Optional: Truncate comment if needed

        logger.info(
            "Successfully parsed Anthropic response for student %s: Score %d",
            student_id,
            score,
        )
        return LLMOutput(
            student_id=student_id, # Use original student_id
            score=score,
            comment=comment,
            raw_response=raw_text,
        )
      else:
        logger.error(
            "Invalid Anthropic response format for student %s: Expected 'id,score,comment', got: %s",
            student_id,
            cleaned_text,
        )
        return LLMOutput(
            student_id=student_id,
            error=f"Invalid response format: {cleaned_text}",
            raw_response=raw_text,
        )
    except (AttributeError, IndexError, Exception) as e:
      logger.exception(
          "Error parsing Anthropic response for student %s: %s", student_id, e
      )
      return LLMOutput(
          student_id=student_id,
          error=f"Failed to parse response: {e}",
          raw_response=str(response),
      )

  # Inherits @retry from the base class
  async def grade(
      self, system_prompt: str, requirements: str, llm_input: LLMInput
  ) -> LLMOutput:
    """Grades the assignment using the Anthropic API.

    Args:
        system_prompt: The system prompt.
        requirements: The assignment requirements.
        llm_input: The input data.

    Returns:
        An LLMOutput object with the results or error information.
    """
    try:
      payload = await self._prepare_request_payload(
          system_prompt, requirements, llm_input
      )
      logger.info(
          "Sending request to Anthropic model %s for student %s.",
          self.model_name,
          llm_input.student_id,
      )

      response: anthropic.types.Message = await self.client.messages.create(
          **payload
      )

      logger.debug(
          "Received Anthropic response for student %s.", llm_input.student_id
      )
      return await self._parse_response(response, llm_input.student_id)

    except anthropic.APIConnectionError as e:
        logger.error("Anthropic API request failed to connect: %s", e)
        return LLMOutput(student_id=llm_input.student_id, error=f"API Connection Error: {e}")
    except anthropic.RateLimitError as e:
        logger.error("Anthropic API request exceeded rate limit: %s", e)
        return LLMOutput(student_id=llm_input.student_id, error=f"API Rate Limit Error: {e}")
    except anthropic.APIStatusError as e:
        logger.error("Anthropic API returned an error status: %s", e)
        return LLMOutput(student_id=llm_input.student_id, error=f"API Status Error {e.status_code}: {e.response}")
    except Exception as e:
      logger.exception(
          "An unexpected error occurred during Anthropic grading for student %s: %s",
          llm_input.student_id,
          e,
      )
      return LLMOutput(
          student_id=llm_input.student_id,
          error=f"Unexpected error during grading: {e}",
      )