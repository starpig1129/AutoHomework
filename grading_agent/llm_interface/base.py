# -*- coding: utf-8 -*-
"""Defines the base interface and data structures for LLM clients."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


@dataclass
class LLMInput:
  """Data structure for input to the LLM."""

  student_id: str
  student_name: str
  text_content: List[str] = field(default_factory=list)
  image_content: List[str] = field(
      default_factory=list
  )  # Base64 encoded images


@dataclass
class LLMOutput:
  """Data structure for the output from the LLM."""

  student_id: str
  score: int = -1 # Default to -1 to indicate failure or not graded
  comment: str = ""
  raw_response: Any = None
  error: str | None = None # Optional field for error messages


class LLMInterface(ABC):
  """Abstract Base Class for asynchronous LLM clients."""

  @abstractmethod
  def __init__(self, api_key: str, model_name: str, **kwargs: Any):
    """Initializes the LLM client.

    Args:
        api_key: The API key for the LLM service.
        model_name: The specific model to use.
        **kwargs: Additional keyword arguments for the client.
    """
    self.api_key = api_key
    self.model_name = model_name
    logger.info("Initialized LLMInterface with model: %s", model_name)

  @abstractmethod
  async def _prepare_request_payload(
      self, system_prompt: str, requirements: str, llm_input: LLMInput
  ) -> Dict[str, Any]:
    """Prepares the request payload specific to the LLM API.

    Args:
        system_prompt: The system prompt to guide the LLM.
        requirements: The specific assignment requirements.
        llm_input: The standardized input data containing student info and content.

    Returns:
        A dictionary representing the request payload for the LLM API.
    """
    pass

  @abstractmethod
  async def _parse_response(
      self, response: Any, student_id: str
  ) -> LLMOutput:
    """Parses the LLM API response into the standardized LLMOutput format.

    Args:
        response: The raw response object from the LLM API call.
        student_id: The student ID associated with this response.

    Returns:
        An LLMOutput object containing the parsed grading results.
    """
    pass

  # Retry 3 times with a 2-second wait between attempts
  @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
  @abstractmethod
  async def grade(
      self, system_prompt: str, requirements: str, llm_input: LLMInput
  ) -> LLMOutput:
    """Performs the grading task by sending a request to the LLM.

    This method orchestrates the process: preparing the request, sending it
    to the LLM API (implementation-specific), and parsing the response.
    Includes retry logic for transient errors.

    Args:
        system_prompt: The system prompt.
        requirements: The assignment requirements.
        llm_input: The input data for grading.

    Returns:
        An LLMOutput object with the grading results.

    Raises:
        Exception: If the API call fails after retries.
    """
    logger.debug(
        "Starting grade task for student %s with model %s",
        llm_input.student_id,
        self.model_name,
    )
    # Implementation details (API call) will be in the concrete classes.
    # This abstract method defines the signature and includes the retry decorator.
    pass