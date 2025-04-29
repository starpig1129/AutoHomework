# -*- coding: utf-8 -*-
"""Core Grading Agent implementation."""

import asyncio
import logging
import os
import time
from typing import List, Optional, Tuple

from grading_agent.config.config_manager import ConfigManager, get_config_manager
from grading_agent.grading.logic import validate_grading_result
from grading_agent.llm_interface.base import LLMInput, LLMInterface, LLMOutput
from grading_agent.llm_interface.factory import LLMFactory
from grading_agent.output.csv_writer import AsyncCSVWriter
from grading_agent.processors.file_processor import FileProcessor

# Setup basic logging if not configured elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GradingAgent:
    """Coordinates the asynchronous grading process for student assignments."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initializes the GradingAgent.

        Args:
            config_manager: An optional ConfigManager instance. If None, creates one.
        """
        self.config_manager = config_manager or get_config_manager()
        self.llm_config = self.config_manager.get_llm_config()
        self.path_config = self.config_manager.get_path_config()

        # Validate essential paths
        self.homework_dir = self._validate_path(self.path_config.get('homework_dir'), 'homework_dir')
        self.output_csv_path = self.path_config.get('output_csv')
        if not self.output_csv_path:
             raise ValueError("Missing 'output_csv' path in configuration.")

        self.system_prompt_path = self.path_config.get('system_prompt_file', 'system_prompt.txt')
        self.requirements_path = self.path_config.get('requirements_file', 'assignment_requirements.txt')

        # Load prompts
        self.system_prompt = self._load_prompt_file(self.system_prompt_path)
        self.assignment_requirements = self._load_prompt_file(self.requirements_path)

        # Initialize components
        self.file_processor = FileProcessor(
            max_depth=self.config_manager.get('processing.max_depth', 5),
            max_zip_size_mb=self.config_manager.get('processing.max_zip_size_mb', 100)
        )
        self.llm_client = self._create_llm_client()
        self.csv_writer = AsyncCSVWriter(self.output_csv_path)

        logger.info("GradingAgent initialized successfully.")

    def _validate_path(self, path: Optional[str], name: str) -> str:
        """Validates if a required path exists and is a directory."""
        if not path:
            raise ValueError(f"Missing '{name}' path in configuration.")
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            raise FileNotFoundError(f"Configured '{name}' directory not found or is not a directory: {abs_path}")
        logger.info("Validated path for %s: %s", name, abs_path)
        return abs_path

    def _load_prompt_file(self, file_path: str) -> str:
        """Loads content from a prompt file."""
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
             logger.warning("Prompt file not found: %s. Using empty string.", abs_path)
             return ""
        try:
             with open(abs_path, 'r', encoding='utf-8') as f:
                 content = f.read()
             logger.info("Loaded prompt from: %s", abs_path)
             return content
        except Exception as e:
             logger.error("Failed to read prompt file %s: %s", abs_path, e)
             return "" # Return empty on error

    def _create_llm_client(self) -> LLMInterface:
        """Creates the LLM client instance using the factory.

        Reads the provider configuration, retrieves the API key from the
        environment variable specified in the config, and instantiates the client.

        Returns:
            An instance of LLMInterface for the configured provider.

        Raises:
            ValueError: If required configuration (provider, model, api_key_env)
                        is missing or the API key environment variable is not set.
            NotImplementedError: If the specified provider is not supported by the factory.
        """
        # Retrieve LLM configuration using dedicated methods
        provider = self.config_manager.get_llm_provider()
        model_name = self.config_manager.get_llm_model()
        provider_config = self.config_manager.get_llm_config() # Still needed for api_key_env and parameters
        api_key_env_var = provider_config.get('api_key_env') # Get the env var name from the specific provider config

        # Log the retrieved provider and model for debugging
        logger.debug("Provider from ConfigManager: %s", provider)
        logger.debug("Model from ConfigManager: %s", model_name)

        if not provider or not model_name:
            logger.error("LLM 'provider' or 'model' name missing or empty in configuration.") # Log before raising
            raise ValueError("LLM 'provider' or 'model' name missing in configuration.")
        if not api_key_env_var:
            # Log before raising
            logger.error("LLM configuration for provider '%s' is missing the 'api_key_env' setting.", provider)
            raise ValueError(f"LLM configuration for provider '{provider}' is missing the 'api_key_env' setting.")

        # Get API key from environment using the name specified in config
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(f"API key environment variable '{api_key_env_var}' not set for provider '{provider}'.")

        # Pass other potential LLM args from config (e.g., max_tokens, temperature)
        llm_args = provider_config.get('parameters', {})

        try:
            client = LLMFactory.create_llm_client(
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                **llm_args
            )
            logger.info("LLM client created for provider '%s', model '%s'.", provider, model_name)
            return client
        except (ValueError, NotImplementedError) as e:
             logger.error("Failed to create LLM client: %s", e)
             raise # Re-raise configuration/factory errors


    async def _process_single_student(self, student_id: str, student_name: str, student_dir: str) -> LLMOutput:
        """Processes assignments for a single student asynchronously.

        Returns:
            LLMOutput: Contains grading result or error information.
                       Specific error strings are used for different failure modes.
        """
        logger.info("Processing student: %s (%s)", student_id, student_name)
        start_time = time.monotonic()

        try:
            # 1. Extract content
            texts, images = await self.file_processor.get_student_content(student_dir)

            if not texts and not images:
                logger.warning("No content found for student %s in %s.", student_id, student_dir)
                # Return specific error for "No Submission"
                # student_name is handled in the run loop when writing CSV
                return LLMOutput(student_id=student_id, score=0, comment="未繳交", error="No submission: No processable files found.")

            # 2. Prepare LLM Input
            # Assuming LLMInput takes student_name (modify if needed)
            llm_input = LLMInput(
                student_id=student_id,
                student_name=student_name,
                text_content=texts,
                image_content=images
            )

            # 3. Call LLM for grading
            logger.debug("Sending grading request for student %s", student_id)
            llm_output = await self.llm_client.grade(
                system_prompt=self.system_prompt,
                requirements=self.assignment_requirements,
                llm_input=llm_input
            )
            logger.debug("Received grading response for student %s", student_id)

            # Handle case where LLM returns None or empty response
            if llm_output is None:
                 logger.error("LLM client returned None for student %s.", student_id)
                 # student_name is handled in the run loop when writing CSV
                 return LLMOutput(student_id=student_id, score=0, comment="評分異常", error="LLM error: No response received.")

            # Ensure student_name is populated in the successful output
            # Assuming LLMOutput has student_name field (modify if needed)
            llm_output.student_id = student_id # Ensure ID/Name are consistent
            llm_output.student_name = student_name

            # 4. Validate result
            if not validate_grading_result(llm_output):
                 logger.error("Grading result validation failed for student %s. Raw response: %s",
                              student_id, llm_output.raw_response)
                 # Return specific error for "Grading Error / Validation Failed"
                 # student_name is handled in the run loop when writing CSV
                 return LLMOutput(student_id=student_id, score=0, comment="評分異常", error=f"Grading error: Validation failed. Raw: {llm_output.raw_response[:100]}")

            end_time = time.monotonic()
            logger.info("Successfully processed student %s in %.2f seconds. Score: %s", # Use %s for score flexibility
                        student_id, end_time - start_time, llm_output.score)
            return llm_output

        except FileNotFoundError as e: # More specific error for file issues
            end_time = time.monotonic()
            logger.exception("File processing error for student %s (%s) after %.2f seconds: %s",
                             student_id, student_name, end_time - start_time, e)
            # student_name is handled in the run loop when writing CSV
            return LLMOutput(student_id=student_id, score=0, comment="讀取異常", error=f"File processing error: {e}")
        except asyncio.TimeoutError as e: # Specific handling for timeouts (likely LLM)
            end_time = time.monotonic()
            logger.exception("LLM timeout error for student %s (%s) after %.2f seconds: %s",
                             student_id, student_name, end_time - start_time, e)
            # student_name is handled in the run loop when writing CSV
            return LLMOutput(student_id=student_id, score=0, comment="評分異常", error=f"LLM error: Timeout - {e}")
        except Exception as e: # Catch other potential errors (LLM client connection, validation logic, etc.)
            end_time = time.monotonic()
            # Try to determine if it's more likely a grading or processing error
            error_context = str(e).lower()
            if "llm" in error_context or "api" in error_context or "grade" in error_context:
                error_type = "Grading error"
                comment = "評分異常"
            else:
                error_type = "Agent processing error"
                comment = "讀取異常" # Default to read error for unexpected agent issues

            logger.exception("%s for student %s (%s) after %.2f seconds: %s",
                             error_type.capitalize(), student_id, student_name, end_time - start_time, e)
            # student_name is handled in the run loop when writing CSV
            return LLMOutput(student_id=student_id, score=0, comment=comment, error=f"{error_type}: {e}")


    async def run(self, max_concurrent_students: int = 5):
        """Runs the grading process for all students in the homework directory."""
        logger.info("Starting Grading Agent run...")
        overall_start_time = time.monotonic()

        # Initialize CSV writer (overwrite existing file)
        await self.csv_writer.initialize_file(overwrite=True)
        logger.info("Initialized CSV writer for %s", self.output_csv_path) # Added log

        student_tasks = [] # List to hold the asyncio tasks
        processed_count = 0
        failed_count = 0
        skipped_count = 0 # For folders not matching format or explicitly skipped
        no_submission_count = 0
        read_error_count = 0
        grading_error_count = 0

        try:
            logger.info("Scanning homework directory: %s", self.homework_dir) # Added log
            # Scan homework directory for student folders (assuming 'id-name' format)
            # Use sync os.listdir for initial scan, async processing for contents
            try: # Added inner try block for listdir
                entries = os.listdir(self.homework_dir)
                logger.info("Found %d entries in homework directory: %s", len(entries), entries) # Added log
            except OSError as e:
                logger.error("Failed to list homework directory '%s': %s", self.homework_dir, e)
                entries = [] # Continue with empty list if listing fails

            # Define the wrapper coroutine to return student info alongside the result
            async def process_and_wrap(student_id, student_name, student_dir):
                result = await self._process_single_student(student_id, student_name, student_dir)
                return student_id, student_name, result # Return student info along with result

            for entry in entries:
                logger.debug("Processing entry: %s", entry)
                student_dir = os.path.join(self.homework_dir, entry)
                if os.path.isdir(student_dir):
                    logger.debug("Entry is a directory: %s", student_dir)
                    # Attempt to parse ID and Name from folder name (using '_' as separator)
                    parts = entry.split('_', 1)
                    if len(parts) == 2:
                        student_id = parts[0].strip()
                        # Take only the part before potential suffixes like '_無附件'
                        student_name = parts[1].split('_', 1)[0].strip()
                        if student_id and student_name: # Basic validation
                            logger.info("Creating task for student ID: %s, Name: %s, Dir: %s", student_id, student_name, student_dir)
                            # Create task using the wrapper
                            task = asyncio.create_task(
                                process_and_wrap(student_id, student_name, student_dir)
                            )
                            student_tasks.append(task)
                        else:
                            logger.warning("Skipping directory: Invalid ID or Name after parsing '%s' with '_': ID='%s', Name='%s'",
                                           entry, parts[0], parts[1])
                            skipped_count += 1
                    else:
                        logger.warning("Skipping directory: Name format mismatch ('id_name' expected): %s", entry)
                        skipped_count += 1
                elif os.path.isfile(student_dir):
                    logger.debug("Skipping file entry: %s", entry)
                    skipped_count += 1
                else:
                    logger.debug("Skipping other non-directory entry: %s", entry)
                    skipped_count += 1

            logger.info("Finished scanning directory. Created %d student processing tasks.", len(student_tasks)) # Added log
            if not student_tasks:
                 logger.warning("No valid student directories found or processed. Check homework_dir contents and directory naming format ('id-name').") # Added warning

            # Process tasks with concurrency limit using a semaphore
            semaphore = asyncio.Semaphore(max_concurrent_students)

            async def run_with_semaphore(task):
                async with semaphore:
                    return await task

            # Use asyncio.gather to wait for all tasks while respecting concurrency
            # results_data = await asyncio.gather(*[run_with_semaphore(task) for task in student_tasks])
            # Using as_completed to process results as they arrive and write immediately
            logger.info("Processing %d tasks with concurrency limit %d...", len(student_tasks), max_concurrent_students)
            futures = [run_with_semaphore(task) for task in student_tasks]
            for future in asyncio.as_completed(futures):
                try:
                    student_id, student_name, result = await future # Unpack the result from the wrapper

                    if result is None: # Should not happen with new logic, but handle defensively
                        logger.error("Internal error: process_and_wrap returned None result for %s (%s). Skipping write.", student_id, student_name)
                        failed_count += 1 # Count as failure
                        grading_error_count += 1 # Attribute to grading/agent error
                        continue

                    if result.error:
                        failed_count += 1
                        error_message = result.error.lower()
                        score = 0 # Default score for errors

                        # Determine comment and specific error count based on error type
                        if "no submission" in error_message:
                            comment = "未繳交"
                            no_submission_count += 1
                        elif "file processing error" in error_message:
                            comment = "讀取異常"
                            read_error_count += 1
                        elif "llm error" in error_message or "grading error" in error_message or "validation failed" in error_message:
                            comment = "評分異常"
                            grading_error_count += 1
                        elif "agent processing error" in error_message: # Catch-all for other agent issues
                            comment = "讀取異常" # Treat unexpected agent errors as read errors
                            read_error_count += 1
                        else:
                            comment = f"處理失敗: {result.error[:50]}" # Generic fallback
                            logger.warning("Unhandled error type for %s: %s. Using generic comment.", student_id, result.error)
                            # Decide how to categorize unknown errors, maybe grading?
                            grading_error_count += 1

                        await self.csv_writer.write_result(student_id, student_name, score, comment)
                        logger.error("Failed processing for %s (%s): %s", student_id, student_name, result.error)

                    else: # Success case
                        processed_count += 1
                        await self.csv_writer.write_result(
                            student_id=result.student_id,
                            student_name=result.student_name,
                            score=result.score,
                            comment=result.comment
                        )
                        # No need for extra log here, _process_single_student already logs success

                except Exception as e:
                    # This catches errors in the process_and_wrap, semaphore logic, or task cancellation
                    # We might not have student_id/name here if the wrapper failed early
                    logger.exception("A student processing task wrapper failed unexpectedly: %s", e)
                    failed_count += 1
                    # Cannot reliably write to CSV without student_id/name. Count as grading error?
                    grading_error_count += 1

            # No final batch write needed as results are written individually

        except Exception as e:
            logger.exception("An critical error occurred during the main agent run loop: %s", e)
        finally:
            overall_end_time = time.monotonic()
            total_attempted = len(student_tasks)
            total_written = processed_count + no_submission_count + read_error_count + grading_error_count
            # Note: failed_count might differ from sum of error counts if wrapper itself failed

            logger.info("="*30 + " Grading Run Summary " + "="*30)
            logger.info("Grading Agent run finished in %.2f seconds.", overall_end_time - overall_start_time)
            logger.info("Directory Scan: Found %d entries, Skipped %d (format/file), Attempted %d tasks.",
                        len(entries) if 'entries' in locals() else 'N/A', skipped_count, total_attempted)
            logger.info("Processing Results:")
            logger.info("  - Successfully Graded: %d", processed_count)
            logger.info("  - Failed - No Submission: %d", no_submission_count)
            logger.info("  - Failed - Read Error: %d", read_error_count)
            logger.info("  - Failed - Grading Error: %d", grading_error_count)
            # logger.info("  - Total Failed (incl. task errors): %d", failed_count) # Can be confusing
            logger.info("Total results written to CSV: %d", total_written)
            if total_attempted != total_written:
                 logger.warning("Mismatch between attempted tasks (%d) and written results (%d). Some tasks might have failed critically.",
                                total_attempted, total_written)
            logger.info("Results saved to: %s", self.output_csv_path)
            logger.info("="*80)