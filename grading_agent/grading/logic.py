# -*- coding: utf-8 -*-
"""Contains logic related to grading validation and potential future rubric application."""

import logging
from typing import Optional

from grading_agent.llm_interface.base import LLMOutput

logger = logging.getLogger(__name__)


def validate_grading_result(output: LLMOutput, max_comment_len: Optional[int] = 30) -> bool:
    """
    Performs validation on the parsed LLM output.

    Checks for client errors, valid score range, and non-empty comment.
    Optionally checks comment length.

    Args:
        output: The LLMOutput object from the LLM client.
        max_comment_len: Optional maximum allowed length for the comment.

    Returns:
        True if the output passes validation checks, False otherwise.
    """
    if output.error:
        logger.warning(
            "Validation failed for student %s due to client-reported error: %s",
            output.student_id,
            output.error,
        )
        return False

    # Check score type and range
    if not isinstance(output.score, int) or not (0 <= output.score <= 100):
        logger.warning(
            "Validation failed for student %s: Invalid score '%s' (type: %s). Expected int 0-100.",
            output.student_id,
            output.score,
            type(output.score).__name__,
        )
        return False

    # Check comment type and presence
    if not isinstance(output.comment, str) or not output.comment.strip():
        logger.warning(
            "Validation failed for student %s: Comment is missing or empty.",
            output.student_id,
        )
        # Depending on strictness, decide if empty comment is failure. Assuming it is for now.
        return False

    # Optional: Check comment length
    if max_comment_len is not None and len(output.comment) > max_comment_len:
        logger.warning(
            "Validation warning for student %s: Comment length (%d) exceeds limit (%d). Comment: '%s'",
            output.student_id,
            len(output.comment),
            max_comment_len,
            output.comment[:max_comment_len+20] + "..." # Log truncated comment
        )
        # Decide if this is a hard failure. For now, just a warning, return True.
        # return False # Uncomment if exceeding length should cause failure.

    # --- Start: Preliminary Logic Validation ---
    # Note: This is a basic check. A more robust implementation might involve
    # sentiment analysis or keyword spotting for positive/negative comments.
    # Currently, it only logs warnings, doesn't cause validation failure.

    # Example keywords (can be expanded)
    negative_keywords = ["錯誤", "不完整", "缺失", "需要改進", "不及格"]
    positive_keywords = ["優秀", "完美", "正確", "清晰", "很好"]

    # Check 1: High score should generally not have a purely negative comment.
    if output.score > 90:
        # Use the defined negative_keywords
        if any(keyword in output.comment for keyword in negative_keywords):
            logger.warning(
                "Validation warning for student %s: High score (%d) but potentially negative comment: '%s'",
                output.student_id,
                output.score,
                output.comment
            )
            # Decide if this should be a failure. For now, just a warning.

    # Check 2: Low score should generally not have a purely positive comment.
    if output.score < 50:
        # Use the defined positive_keywords and negative_keywords
        # Check if comment seems purely positive (this is a simplification)
        is_potentially_positive = any(keyword in output.comment for keyword in positive_keywords)
        is_potentially_negative = any(keyword in output.comment for keyword in negative_keywords) # Reuse negative keywords
        if is_potentially_positive and not is_potentially_negative: # Simple check: positive keywords without negative ones
             logger.warning(
                "Validation warning for student %s: Low score (%d) but potentially positive comment: '%s'",
                output.student_id,
                output.score,
                output.comment
            )
            # Decide if this should be a failure. For now, just a warning.
    # --- End: Preliminary Logic Validation ---


    logger.debug("Validation passed for student %s (Score: %d)", output.student_id, output.score)
    return True


# Future functions can be added here, for example:
# def apply_grading_rubric(output: LLMOutput, rubric: Dict) -> LLMOutput:
#     """Applies specific grading rules based on a rubric."""
#     # ... implementation ...
#     pass