# -*- coding: utf-8 -*-
"""Implementation of the LLMInterface for Google Gemini models."""

import base64
import io
import logging
from typing import Any, Dict, List

import google.generativeai as genai
from google.generativeai import types as generation_types
from PIL import Image

from .base import LLMInput, LLMInterface, LLMOutput

logger = logging.getLogger(__name__)


class GeminiClient(LLMInterface):
  """Concrete implementation of LLMInterface for Google Gemini models."""

  def __init__(self, api_key: str, model_name: str, **kwargs: Any):
    """Initializes the Google Gemini client.

    Args:
        api_key: The Google API key.
        model_name: The specific Gemini model to use (e.g., "gemini-1.5-flash").
        **kwargs: Additional keyword arguments for the Gemini client.
                  Common arguments include 'max_output_tokens', 'temperature',
                  'top_p', 'top_k'.
    """
    super().__init__(api_key=api_key, model_name=model_name, **kwargs)
    try:
      genai.configure(api_key=self.api_key)
      # Extract generation config parameters from kwargs
      self.generation_config = generation_types.GenerationConfig(
          # candidate_count=kwargs.get('candidate_count', 1), # Usually 1
          # stop_sequences=kwargs.get('stop_sequences', None),
          max_output_tokens=kwargs.get("max_output_tokens", 300),
          temperature=kwargs.get("temperature", 0.0),
          top_p=kwargs.get("top_p", None),
          top_k=kwargs.get("top_k", None),
      )
      self.model = genai.GenerativeModel(self.model_name)
      logger.info(
          "Gemini client initialized successfully for model %s.", self.model_name
      )
    except Exception as e:
      logger.exception(
          "Failed to initialize Gemini client for model %s: %s",
          self.model_name,
          e,
      )
      raise

  async def _prepare_request_payload(
      self, system_prompt: str, requirements: str, llm_input: LLMInput
  ) -> Dict[str, Any]:
    """Prepares the request payload for the Gemini API.

    Args:
        system_prompt: The system prompt (used as initial text part).
        requirements: The assignment requirements.
        llm_input: The standardized input data.

    Returns:
        A dictionary containing 'contents' and 'generation_config'.
    """
    contents: List[Any] = []

    # Combine system prompt and requirements as the initial user turn
    initial_prompt = f"{system_prompt}\n\n作業要求：\n{requirements}\n\n學生資訊：{llm_input.student_id} - {llm_input.student_name}"
    contents.append(initial_prompt)

    # Add text content if available
    if llm_input.text_content:
      full_text = "作業文字內容：\n" + "\n".join(llm_input.text_content)
      contents.append(full_text)
    else:
      contents.append("無文字內容提交。")

    # Add image content if available
    if llm_input.image_content:
      contents.append("作業圖片內容如下：")
      for img_base64 in llm_input.image_content:
        # Remove potential data URI prefix
        if "base64," in img_base64:
          img_base64 = img_base64.split("base64,")[-1]
        try:
          # Decode base64 and create PIL Image object
          img_bytes = base64.b64decode(img_base64)
          img = Image.open(io.BytesIO(img_bytes))
          # Gemini SDK prefers PIL Images directly for some models/versions
          # Or use mime_type/data format:
          # contents.append({'mime_type': 'image/jpeg', 'data': img_base64})
          contents.append(img)
        except Exception as e:
          logger.warning(
              "Failed to decode or load image for student %s: %s. Skipping image.",
              llm_input.student_id,
              e,
          )
          contents.append("[圖片處理失敗]")
    else:
      contents.append("無圖片內容提交。")

    # Final instruction
    contents.append("請評分這位學生的作業。")

    payload = {
        "contents": contents,
        "generation_config": self.generation_config,
        # stream=False # Default is False for generate_content_async
    }
    logger.debug(
        "Prepared Gemini request payload for student %s.", llm_input.student_id
    )
    # Avoid logging full payload if images are large
    # logger.debug("Payload contents (text parts): %s", [p for p in contents if isinstance(p, str)])
    return payload

  async def _parse_response(
      self, response: generation_types.GenerateContentResponse, student_id: str
  ) -> LLMOutput:
    """Parses the Gemini API response.

    Args:
        response: The GenerateContentResponse object from the Gemini API.
        student_id: The student ID.

    Returns:
        An LLMOutput object.
    """
    try:
      # Check for safety blocks or empty candidates
      if not response.candidates:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        error_msg = f"No candidates returned. Prompt Feedback: {prompt_feedback}"
        logger.warning(
            "Gemini response issue for student %s: %s", student_id, error_msg
        )
        return LLMOutput(student_id=student_id, error=error_msg, raw_response=str(response))

      # Check safety ratings if needed (optional)
      # safety_ratings = getattr(response.candidates[0], 'safety_ratings', [])
      # if any(rating.category != safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED and rating.probability != safety_types.HarmProbability.NEGLIGIBLE for rating in safety_ratings):
      #     logger.warning("Potential safety issue detected for student %s: %s", student_id, safety_ratings)
      #     # Decide how to handle safety issues

      raw_text = response.text
      if not raw_text:
        logger.warning(
            "Received empty response text for student %s.", student_id
        )
        return LLMOutput(
            student_id=student_id,
            error="Empty response text from LLM",
            raw_response=str(response),
        )

      # Clean the response: remove potential markdown/quotes
      cleaned_text = raw_text.strip().strip('"`').strip()
      logger.info(
          "Raw Gemini response for student %s: %s", student_id, cleaned_text
      )

      parts = [p.strip() for p in cleaned_text.split(',')]
      if len(parts) >= 3:
        parsed_student_id = parts[0]
        # Optional: Validate student ID if needed
        # if parsed_student_id != student_id:
        #     logger.warning("Mismatched student ID in response for %s: got %s", student_id, parsed_student_id)

        try:
          score = int(parts[1])
          if not 0 <= score <= 100:
            logger.warning(
                "Score out of range for student %s: %d. Clamping.",
                student_id,
                score,
            )
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
        # Optional: Truncate comment if needed

        logger.info(
            "Successfully parsed Gemini response for student %s: Score %d",
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
            "Invalid Gemini response format for student %s: Expected 'id,score,comment', got: %s",
            student_id,
            cleaned_text,
        )
        return LLMOutput(
            student_id=student_id,
            error=f"Invalid response format: {cleaned_text}",
            raw_response=raw_text,
        )
    except (AttributeError, IndexError, ValueError, Exception) as e:
      logger.exception(
          "Error parsing Gemini response for student %s: %s", student_id, e
      )
      return LLMOutput(
          student_id=student_id,
          error=f"Failed to parse response: {e}",
          raw_response=str(response),
      )

  # The @retry decorator is inherited from the base class method
  async def grade(
      self, system_prompt: str, requirements: str, llm_input: LLMInput
  ) -> LLMOutput:
    """Grades the assignment using the Gemini API.

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
          "Sending request to Gemini model %s for student %s.",
          self.model_name,
          llm_input.student_id,
      )

      response: generation_types.GenerateContentResponse = (
          await self.model.generate_content_async(**payload)
      )

      logger.debug(
          "Received Gemini response for student %s.", llm_input.student_id
      )
      return await self._parse_response(response, llm_input.student_id)

    except generation_types.BlockedPromptException as e:
        logger.error("Gemini API request failed due to blocked prompt for student %s: %s", llm_input.student_id, e)
        return LLMOutput(student_id=llm_input.student_id, error=f"Blocked Prompt Error: {e}", raw_response=str(e))
    except generation_types.StopCandidateException as e:
         logger.error("Gemini API request failed due to stopped candidate for student %s: %s", llm_input.student_id, e)
         # The partial response might be in e.response
         raw_resp = getattr(e, 'response', str(e))
         return LLMOutput(student_id=llm_input.student_id, error=f"Stopped Candidate Error: {e}", raw_response=str(raw_resp))
    except Exception as e:
      # Catch errors during payload prep, API call, or parsing
      logger.exception(
          "An unexpected error occurred during Gemini grading for student %s: %s",
          llm_input.student_id,
          e,
      )
      return LLMOutput(
          student_id=llm_input.student_id,
          error=f"Unexpected error during grading: {e}",
      )