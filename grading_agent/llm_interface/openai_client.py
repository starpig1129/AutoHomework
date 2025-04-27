# -*- coding: utf-8 -*-
"""Implementation of the LLMInterface for OpenAI models."""

import logging
from typing import Any, Dict, List

import openai
from openai.types.chat import ChatCompletion

from .base import LLMInput, LLMInterface, LLMOutput

logger = logging.getLogger(__name__)


class OpenAIClient(LLMInterface):
  """Concrete implementation of LLMInterface for OpenAI models using async API."""

  def __init__(self, api_key: str, model_name: str, **kwargs: Any):
    """Initializes the asynchronous OpenAI client.

    Args:
        api_key: The OpenAI API key.
        model_name: The specific OpenAI model to use (e.g., "gpt-4o-mini").
        **kwargs: Additional keyword arguments for the OpenAI client.
                  Common arguments include 'max_tokens', 'temperature'.
    """
    super().__init__(api_key=api_key, model_name=model_name, **kwargs)
    try:
      self.client = openai.AsyncOpenAI(api_key=self.api_key)
      self.max_tokens = kwargs.get("max_tokens", 300) # Default from main.py
      self.temperature = kwargs.get("temperature", 0.0) # Default from main.py
      logger.info(
          "AsyncOpenAI client initialized successfully for model %s.",
          self.model_name,
      )
    except Exception as e:
      logger.exception(
          "Failed to initialize AsyncOpenAI client for model %s: %s",
          self.model_name,
          e,
      )
      raise

  async def _prepare_request_payload(
      self, system_prompt: str, requirements: str, llm_input: LLMInput
  ) -> Dict[str, Any]:
    """Prepares the request payload for the OpenAI Chat Completions API.

    Args:
        system_prompt: The system prompt.
        requirements: The assignment requirements.
        llm_input: The standardized input data.

    Returns:
        A dictionary representing the request payload.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"作業要求：\n{requirements}"},
        {
            "role": "user",
            "content": f"學生資訊：{llm_input.student_id} - {llm_input.student_name}",
        },
    ]

    # Combine text and image content for the user message
    user_content: List[Dict[str, Any]] = []

    # Add text parts first
    if llm_input.text_content:
      full_text = "作業文字內容：\n" + "\n".join(llm_input.text_content)
      user_content.append({"type": "text", "text": full_text})
    else:
       user_content.append({"type": "text", "text": "無文字內容提交。"})


    # Add image parts
    if llm_input.image_content:
       user_content.append({"type": "text", "text": "作業圖片內容如下："})
       for img_base64 in llm_input.image_content:
          # Ensure base64 string doesn't contain data URI prefix
          if "base64," in img_base64:
              img_base64 = img_base64.split("base64,")[-1]
          user_content.append({
              "type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
          })
    else:
        user_content.append({"type": "text", "text": "無圖片內容提交。"})


    # Add the combined content as a single user message
    messages.append({"role": "user", "content": user_content})

    # Final instruction
    messages.append({"role": "user", "content": "請評分這位學生的作業。"})


    payload = {
        "model": self.model_name,
        "messages": messages,
        "max_tokens": self.max_tokens,
        "temperature": self.temperature,
    }
    logger.debug(
        "Prepared OpenAI request payload for student %s.", llm_input.student_id
    )
    # logger.debug("Payload messages: %s", messages) # Be careful logging potentially large image data
    return payload

  async def _parse_response(
      self, response: ChatCompletion, student_id: str
  ) -> LLMOutput:
    """Parses the OpenAI API response.

    Args:
        response: The ChatCompletion object from the OpenAI API.
        student_id: The student ID.

    Returns:
        An LLMOutput object.
    """
    try:
      raw_text = response.choices[0].message.content
      if not raw_text:
        logger.warning(
            "Received empty response content for student %s.", student_id
        )
        return LLMOutput(
            student_id=student_id, error="Empty response from LLM"
        )

      # Clean the response: remove potential markdown/quotes
      cleaned_text = raw_text.strip().strip('"`').strip()
      logger.info(
          "Raw response for student %s: %s", student_id, cleaned_text
      )

      parts = [p.strip() for p in cleaned_text.split(',')]
      if len(parts) >= 3:
        parsed_student_id = parts[0]
        # Validate student ID if possible/needed here
        # if parsed_student_id != student_id:
        #     logger.warning("Mismatched student ID in response for %s: got %s", student_id, parsed_student_id)
        #     # Decide how to handle mismatch, e.g., use original ID or log error

        try:
          score = int(parts[1])
          if not 0 <= score <= 100:
             logger.warning("Score out of range for student %s: %d", student_id, score)
             # Decide handling: clamp score, return error, etc.
             # Clamping for now:
             score = max(0, min(100, score))

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
        # Optional: Truncate comment if needed, based on requirements
        # max_comment_length = 30 # Example
        # if len(comment) > max_comment_length:
        #    logger.warning("Comment too long for student %s, truncating.", student_id)
        #    comment = comment[:max_comment_length]


        logger.info(
            "Successfully parsed response for student %s: Score %d",
            student_id,
            score,
        )
        return LLMOutput(
            student_id=student_id, # Use original student_id for consistency
            score=score,
            comment=comment,
            raw_response=raw_text,
        )
      else:
        logger.error(
            "Invalid response format for student %s: Expected 'id,score,comment', got: %s",
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
          "Error parsing OpenAI response for student %s: %s", student_id, e
      )
      return LLMOutput(
          student_id=student_id,
          error=f"Failed to parse response: {e}",
          raw_response=str(response), # Log the raw response object on error
      )

  # The @retry decorator is inherited from the base class method
  async def grade(
      self, system_prompt: str, requirements: str, llm_input: LLMInput
  ) -> LLMOutput:
    """Grades the assignment using the OpenAI API.

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
          "Sending request to OpenAI model %s for student %s.",
          self.model_name,
          llm_input.student_id,
      )

      response: ChatCompletion = await self.client.chat.completions.create(
          **payload
      )

      logger.debug(
          "Received OpenAI response for student %s.", llm_input.student_id
      )
      return await self._parse_response(response, llm_input.student_id)

    except openai.APIConnectionError as e:
        logger.error("OpenAI API request failed to connect: %s", e)
        return LLMOutput(student_id=llm_input.student_id, error=f"API Connection Error: {e}")
    except openai.RateLimitError as e:
        logger.error("OpenAI API request exceeded rate limit: %s", e)
        return LLMOutput(student_id=llm_input.student_id, error=f"API Rate Limit Error: {e}")
    except openai.APIStatusError as e:
        logger.error("OpenAI API returned an error status: %s", e)
        return LLMOutput(student_id=llm_input.student_id, error=f"API Status Error {e.status_code}: {e.response}")
    except Exception as e:
      # This will catch errors during payload prep, API call, or parsing
      # The retry decorator handles retries for API call failures based on its configuration.
      logger.exception(
          "An unexpected error occurred during OpenAI grading for student %s: %s",
          llm_input.student_id,
          e,
      )
      # Return an error LLMOutput; the exception might be raised by tenacity if retries fail
      return LLMOutput(
          student_id=llm_input.student_id,
          error=f"Unexpected error during grading: {e}",
      )