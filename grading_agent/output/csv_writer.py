# -*- coding: utf-8 -*-
"""Asynchronous CSV writer for grading results."""

import csv
import logging
import os
import io # Added for StringIO
import asyncio
from typing import List, Sequence, Union # Added Union

try:
    import aiofiles
    import aiofiles.os
except ImportError:
    aiofiles = None # type: ignore

from grading_agent.llm_interface.base import LLMOutput

logger = logging.getLogger(__name__)

class AsyncCSVWriter:
    """Writes grading results to a CSV file asynchronously."""

    def __init__(self, output_path: str):
        """Initializes the AsyncCSVWriter.

        Args:
            output_path: The path to the output CSV file.
        """
        if aiofiles is None:
            logger.warning("aiofiles library not found. CSV writing will be less efficient.")
            # raise ImportError("aiofiles is required for AsyncCSVWriter")
        self.output_path = output_path
        # Updated header to match requirements
        self.header = ['學號', '姓名', '分數', '評語']
        self._initialized = False # Track if header has been written

    async def _ensure_dir_exists(self):
        """Ensure the output directory exists."""
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir): # Check sync first for efficiency
            try:
                if aiofiles:
                    await aiofiles.os.makedirs(output_dir, exist_ok=True)
                else:
                    # Fallback to sync makedirs in thread
                    await asyncio.to_thread(os.makedirs, output_dir, exist_ok=True)
                logger.info("Created output directory: %s", output_dir)
            except Exception as e:
                logger.error("Failed to create output directory %s: %s", output_dir, e)
                raise # Re-raise the error as we cannot proceed

    async def initialize_file(self, overwrite: bool = True):
        """Writes the header row to the CSV file.

        Ensures the output directory exists. Overwrites the file by default.

        Args:
            overwrite: If True, overwrite the file if it exists.
                       If False, checks if header exists or file is empty.
        """
        await self._ensure_dir_exists()
        mode = 'w' if overwrite else 'a+' # Append mode, read/write, create if not exists

        if aiofiles:
            try:
                async with aiofiles.open(self.output_path, mode=mode, newline='', encoding='utf-8') as f:
                    if not overwrite:
                        # Check if file is empty or header already exists
                        await f.seek(0)
                        first_line = await f.readline()
                        # Use csv module to correctly join header with potential commas
                        header_line = io.StringIO()
                        csv.writer(header_line).writerow(self.header)
                        expected_header = header_line.getvalue().strip()
                        if first_line and first_line.strip() == expected_header:
                            logger.info("CSV header already exists in %s. Skipping write.", self.output_path)
                            self._initialized = True
                            return
                        elif first_line:
                            logger.warning("CSV file %s exists but has unexpected header or content ('%s' vs expected '%s'). Appending anyway.",
                                           self.output_path, first_line.strip(), expected_header)
                            # Position at the end for appending
                            await f.seek(0, os.SEEK_END)
                        # If file was empty or we are appending after check, fall through to write header

                    # Write header (either overwrite or append to empty/checked file)
                    # csv.writer expects a file-like object supporting write()
                    # aiofiles file handle should work directly here.
                    # We need to wrap the sync csv writer logic.
                    # Let's try writing directly first. If issues arise, use to_thread.

                    # Create a temporary string buffer to use with csv.writer
                    output_buffer = io.StringIO()
                    writer = csv.writer(output_buffer, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(self.header)
                    await f.write(output_buffer.getvalue())

                    logger.info("Successfully wrote CSV header to %s", self.output_path)
                    self._initialized = True
            except Exception as e:
                logger.error("Failed to write CSV header to %s: %s", self.output_path, e)
                raise
        else:
            # Fallback using sync operations in thread
            logger.warning("Using sync file operations for CSV header due to missing aiofiles.")
            try:
                def write_header_sync():
                    _initialized = False
                    _mode = 'w' if overwrite else 'a+'
                    with open(self.output_path, mode=_mode, newline='', encoding='utf-8') as f_sync:
                        if not overwrite:
                            f_sync.seek(0)
                            f_sync.seek(0)
                            first_line = f_sync.readline()
                            header_line_sync = io.StringIO()
                            csv.writer(header_line_sync).writerow(self.header)
                            expected_header_sync = header_line_sync.getvalue().strip()
                            if first_line and first_line.strip() == expected_header_sync:
                                logger.info("CSV header already exists (sync check).")
                                _initialized = True
                                return _initialized
                            elif first_line:
                                logger.warning("CSV file exists with unexpected header (sync check, '%s' vs expected '%s'). Appending.",
                                               first_line.strip(), expected_header_sync)
                                f_sync.seek(0, os.SEEK_END)

                        writer = csv.writer(f_sync, quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(self.header)
                        logger.info("Successfully wrote CSV header (sync).")
                        _initialized = True
                    return _initialized

                self._initialized = await asyncio.to_thread(write_header_sync)
            except Exception as e:
                logger.error("Failed to write CSV header (sync) to %s: %s", self.output_path, e)
                raise


    async def write_result(self, student_id: str, student_name: str, score: Union[int, float, str], comment: str):
        """Appends a single grading result to the CSV file asynchronously.

        Handles different data types for score and ensures proper formatting.

        Args:
            student_id: The student's ID.
            student_name: The student's name.
            score: The score (can be int, float, or string like 'N/A').
            comment: The comment or status (e.g., '未繳交', '評分異常').

        Raises:
            RuntimeError: If the writer has not been initialized (header not written).
        """
        if not self._initialized:
            # Try to initialize non-destructively if not already done
            logger.warning("CSV writer not initialized. Attempting to initialize without overwriting.")
            # Ensure initialization happens before proceeding
            await self.initialize_file(overwrite=False)
            if not self._initialized:
                # If initialization still failed (e.g., file access error), raise
                raise RuntimeError("CSV writer failed to initialize and cannot write results.")

        # Prepare the row with the required fields
        row_to_write = [student_id, student_name, str(score), comment]

        if aiofiles:
            try:
                async with aiofiles.open(self.output_path, mode='a', newline='', encoding='utf-8') as f:
                    # Use a string buffer for csv.writer
                    output_buffer = io.StringIO()
                    writer = csv.writer(output_buffer, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(row_to_write)
                    await f.write(output_buffer.getvalue())

                logger.debug("Successfully appended result for %s to %s", student_id, self.output_path)
            except Exception as e:
                logger.error("Failed to append result for %s to CSV %s: %s", student_id, self.output_path, e)
                # Consider re-raising or alternative error handling
        else:
            # Fallback using sync operations in thread
            logger.warning("Using sync file operations for appending CSV result for %s.", student_id)
            try:
                def append_result_sync():
                    with open(self.output_path, mode='a', newline='', encoding='utf-8') as f_sync:
                        writer = csv.writer(f_sync, quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(row_to_write)
                await asyncio.to_thread(append_result_sync)
                logger.debug("Successfully appended result (sync) for %s to %s", student_id, self.output_path)
            except Exception as e:
                logger.error("Failed to append result (sync) for %s to CSV %s: %s", student_id, self.output_path, e)


# Example Usage (for testing purposes)
async def _test_writer():
    import tempfile
    import shutil # Import shutil for cleanup
    logging.basicConfig(level=logging.INFO)
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "test_grades.csv")
    writer = AsyncCSVWriter(csv_path)
    print(f"Test CSV path: {csv_path}")

    try:
        # Initialize (overwrite)
        await writer.initialize_file(overwrite=True)
        print("Initialized CSV file.")

        # Write some results using the new method
        await writer.write_result(student_id="B12345678", student_name="陳大文", score=85, comment="Good work")
        await writer.write_result(student_id="B87654321", student_name="林小美", score=92, comment="Excellent!")
        await writer.write_result(student_id="F11111111", student_name="張三", score=0, comment="未繳交")
        await writer.write_result(student_id="R22222222", student_name="李四", score=0, comment="讀取異常")
        await writer.write_result(student_id="D33333333", student_name="王五", score=0, comment="評分異常")
        print("Wrote initial results.")

        # Check content (sync read for simplicity in test)
        with open(csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print("\n--- CSV Content After Initial Write ---")
        print(content)
        print("--------------------------------------")
        assert "B12345678,陳大文,85,Good work" in content
        assert "B87654321,林小美,92,Excellent!" in content
        assert "F11111111,張三,0,未繳交" in content
        assert "R22222222,李四,0,讀取異常" in content
        assert "D33333333,王五,0,評分異常" in content

        # Test initialization without overwrite and append more
        writer_append = AsyncCSVWriter(csv_path)
        await writer_append.initialize_file(overwrite=False) # Should detect header
        print("\nInitialized writer again (append mode).")
        await writer_append.write_result(student_id="M44444444", student_name="趙六", score=78.5, comment="Almost perfect")
        print("Appended one more result.")

        with open(csv_path, 'r', encoding='utf-8') as f:
            content_after_append = f.read()
        print("\n--- CSV Content After Append ---")
        print(content_after_append)
        print("-----------------------------")
        assert "M44444444,趙六,78.5,Almost perfect" in content_after_append
        # Ensure header wasn't written twice
        header_line_check = io.StringIO()
        csv.writer(header_line_check).writerow(writer.header)
        expected_header_check = header_line_check.getvalue().strip()
        assert content_after_append.count(expected_header_check) == 1, \
            f"Header '{expected_header_check}' found multiple times or not at all."


    finally:
        # Clean up
        try:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temp directory: %s", temp_dir)
        except Exception as e:
            logger.error("Error cleaning up temp directory %s: %s", temp_dir, e)

if __name__ == "__main__":
    # Requires aiofiles: pip install aiofiles
    # Requires shutil which is standard library
    if aiofiles:
         print("Running AsyncCSVWriter test...")
         asyncio.run(_test_writer())
         print("AsyncCSVWriter test finished.")
    else:
         print("aiofiles not installed, cannot run test.")