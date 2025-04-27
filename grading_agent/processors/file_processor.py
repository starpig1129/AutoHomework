# -*- coding: utf-8 -*-
"""Asynchronously processes student submission files."""

import asyncio
import logging
import os
import zipfile
import shutil
from typing import List, Tuple, Optional

import magic # python-magic library for MIME type detection
try:
    import aiofiles
    import aiofiles.os
except ImportError:
    aiofiles = None # type: ignore

from grading_agent.llm_interface.base import LLMInput
from .converters import convert_file_content

logger = logging.getLogger(__name__)

# Define supported text/code MIME types (add more as needed)
SUPPORTED_TEXT_MIME_PREFIXES = ('text/',)
SUPPORTED_TEXT_MIME_TYPES = (
    'application/json',
    'application/x-python', # Example for .py files
    'application/javascript', # Example for .js files
    # Add other relevant code/config file types
)

# Define supported "convertible" types (handled by converters)
CONVERTIBLE_MIME_TYPES = (
    'application/x-ipynb+json',
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document', # docx
    'application/vnd.openxmlformats-officedocument.presentationml.presentation', # pptx
)
CONVERTIBLE_EXTENSIONS = ('.ipynb',) # Handle cases where MIME might be generic

# Define supported direct image types
IMAGE_MIME_PREFIXES = ('image/',)


class FileProcessor:
    """Handles scanning directories and processing individual files asynchronously."""

    def __init__(self, max_depth: int = 5, max_zip_size_mb: int = 100):
        """Initializes the FileProcessor.

        Args:
            max_depth: Maximum recursion depth for exploring directories/archives.
            max_zip_size_mb: Maximum total size allowed for extracted zip files (in MB).
        """
        if aiofiles is None:
            logger.warning("aiofiles library not found. File operations will be less efficient.")
            # Consider raising an error if aiofiles is strictly required
            # raise ImportError("aiofiles is required for FileProcessor")
        self.max_depth = max_depth
        self.max_zip_size_bytes = max_zip_size_mb * 1024 * 1024
        self._magic_instance = magic.Magic(mime=True) # Initialize magic instance

    async def _get_mime_type(self, file_path: str) -> Optional[str]:
        """Asynchronously get the MIME type of a file using python-magic."""
        try:
            # Run the synchronous magic call in a thread
            mime_type = await asyncio.to_thread(self._magic_instance.from_file, file_path)
            logger.debug("Detected MIME type for %s: %s", file_path, mime_type)
            return mime_type
        except FileNotFoundError:
            logger.error("File not found while getting MIME type: %s", file_path)
            return None
        except Exception as e:
            logger.error("Error getting MIME type for %s: %s", file_path, e)
            return None

    async def _unzip_file(self, zip_path: str, extract_dir: str, current_depth: int) -> List[str]:
        """Asynchronously extracts a zip file securely and returns paths of extracted files."""
        extracted_files: List[str] = []
        temp_extract_base = os.path.join(extract_dir, f"__temp_extract_{os.path.basename(zip_path)}")

        try:
            # Use asyncio.to_thread for the synchronous zipfile operations
            def unzip_sync():
                files_in_zip: List[str] = []
                total_uncompressed_size = 0
                valid_members: List[zipfile.ZipInfo] = []

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        # Security Check 1: Check for path traversal and total size
                        for member in zip_ref.infolist():
                            # Skip directories explicitly
                            if member.is_dir():
                                continue
                            # Check for absolute paths or '..'
                            if member.filename.startswith('/') or '..' in member.filename:
                                logger.warning("Skipping potentially unsafe path in zip %s: %s", zip_path, member.filename)
                                continue
                            # Check size
                            total_uncompressed_size += member.file_size
                            if total_uncompressed_size > self.max_zip_size_bytes:
                                logger.error("Zip file %s exceeds max uncompressed size limit (%d MB). Aborting extraction.",
                                             zip_path, self.max_zip_size_bytes / (1024*1024))
                                return [] # Abort
                            valid_members.append(member)

                        # Security Check 2: Extract to a temporary location first
                        os.makedirs(temp_extract_base, exist_ok=True)
                        logger.info("Extracting %d valid members from %s to %s", len(valid_members), zip_path, temp_extract_base)

                        for member in valid_members:
                            try:
                                # Extract safely
                                zip_ref.extract(member, path=temp_extract_base)
                                extracted_path = os.path.join(temp_extract_base, member.filename)
                                # Normalize path separators
                                extracted_path = os.path.normpath(extracted_path)

                                # Security Check 3: Ensure extracted file is within the temp dir
                                common_prefix = os.path.commonpath([temp_extract_base, extracted_path])
                                if common_prefix != os.path.abspath(temp_extract_base):
                                     logger.warning("Skipping file extracted outside target dir: %s", extracted_path)
                                     # Attempt to remove potentially malicious file
                                     if os.path.exists(extracted_path): os.remove(extracted_path)
                                     continue

                                # Move valid files to the final destination (overwrite if exists)
                                final_path = os.path.join(extract_dir, os.path.basename(member.filename))
                                try:
                                     # Use shutil.move for atomicity if possible, handle potential errors
                                     shutil.move(extracted_path, final_path)
                                     files_in_zip.append(final_path)
                                     logger.debug("Moved extracted file to %s", final_path)
                                except Exception as move_err:
                                     logger.error("Failed to move extracted file %s to %s: %s", extracted_path, final_path, move_err)
                                     # Clean up the extracted file if move failed
                                     if os.path.exists(extracted_path): os.remove(extracted_path)

                            except Exception as extract_err:
                                logger.error("Error extracting member %s from %s: %s", member.filename, zip_path, extract_err)
                                continue # Skip problematic member

                except zipfile.BadZipFile:
                    logger.error("Invalid or corrupted zip file: %s", zip_path)
                    return []
                except FileNotFoundError:
                    logger.error("Zip file not found during sync extraction: %s", zip_path)
                    return []
                except Exception as e:
                    logger.error("Unexpected error during sync zip extraction of %s: %s", zip_path, e)
                    return []
                finally:
                    # Clean up the temporary extraction directory regardless of success/failure
                    if os.path.exists(temp_extract_base):
                        try:
                            shutil.rmtree(temp_extract_base)
                            logger.debug("Cleaned up temporary zip extraction directory: %s", temp_extract_base)
                        except Exception as clean_err:
                            logger.error("Failed to clean up temp zip directory %s: %s", temp_extract_base, clean_err)

                return files_in_zip

            extracted_sync = await asyncio.to_thread(unzip_sync)

            # Process the extracted files recursively if depth allows
            if current_depth < self.max_depth:
                tasks = [
                    self.process_entry(
                        file, current_depth + 1
                    ) for file in extracted_sync
                ]
                results = await asyncio.gather(*tasks)
                # Flatten the results (list of tuples)
                for res_texts, res_images in results:
                    extracted_files.extend(res_texts) # Assuming process_entry returns (texts, images)
                    extracted_files.extend(res_images) # We just need the paths here for further processing maybe?
                    # Let's rethink this - _unzip_file should just return paths
                    # The caller (process_entry) will handle processing them.
                    # So, just return extracted_sync paths.
            else:
                 logger.warning("Max recursion depth reached during zip extraction for %s. Skipping further processing of extracted files.", zip_path)

            return extracted_sync # Return the paths of successfully extracted and moved files

        except Exception as e:
            logger.exception("Error during async zip processing of %s: %s", zip_path, e)
            # Ensure cleanup happens even if async wrapper fails
            if os.path.exists(temp_extract_base):
                 try:
                     await asyncio.to_thread(shutil.rmtree, temp_extract_base)
                 except Exception as clean_err:
                     logger.error("Failed to clean up temp zip directory %s on error: %s", temp_extract_base, clean_err)
            return []


    async def process_entry(self, entry_path: str, current_depth: int) -> Tuple[List[str], List[str]]:
        """Processes a single directory entry (file or subdirectory) asynchronously.

        Args:
            entry_path: The absolute path to the file or directory.
            current_depth: The current recursion depth.

        Returns:
            A tuple containing (list of extracted text content, list of base64 images).
        """
        if current_depth > self.max_depth:
            logger.warning("Max recursion depth (%d) reached. Skipping: %s", self.max_depth, entry_path)
            return [], []

        all_texts: List[str] = []
        all_images: List[str] = []

        try:
            if aiofiles and await aiofiles.os.path.isdir(entry_path):
                logger.debug("Processing directory: %s at depth %d", entry_path, current_depth)
                # Use aiofiles.os.listdir if available
                try:
                    entries = await aiofiles.os.listdir(entry_path)
                    tasks = [
                        self.process_entry(
                            os.path.join(entry_path, sub_entry), current_depth + 1
                        ) for sub_entry in entries
                    ]
                    results = await asyncio.gather(*tasks)
                    for texts, images in results:
                        all_texts.extend(texts)
                        all_images.extend(images)
                except Exception as list_err:
                     logger.error("Error listing directory %s: %s", entry_path, list_err)

            elif os.path.isdir(entry_path): # Fallback for non-aiofiles
                 logger.debug("Processing directory (sync fallback): %s at depth %d", entry_path, current_depth)
                 try:
                     entries = await asyncio.to_thread(os.listdir, entry_path)
                     tasks = [
                         self.process_entry(
                             os.path.join(entry_path, sub_entry), current_depth + 1
                         ) for sub_entry in entries
                     ]
                     results = await asyncio.gather(*tasks)
                     for texts, images in results:
                         all_texts.extend(texts)
                         all_images.extend(images)
                 except Exception as list_err:
                     logger.error("Error listing directory %s (sync fallback): %s", entry_path, list_err)


            elif (aiofiles and await aiofiles.os.path.isfile(entry_path)) or \
                 (not aiofiles and await asyncio.to_thread(os.path.isfile, entry_path)):
                logger.debug("Processing file: %s at depth %d", entry_path, current_depth)
                file_size = 0
                try:
                    if aiofiles:
                        stat_res = await aiofiles.os.stat(entry_path)
                        file_size = stat_res.st_size
                    else:
                        file_size = await asyncio.to_thread(os.path.getsize, entry_path)

                    if file_size == 0:
                        logger.warning("Skipping empty file: %s", entry_path)
                        return [], []
                except FileNotFoundError:
                     logger.error("File not found during size check: %s", entry_path)
                     return [], []
                except Exception as stat_err:
                     logger.error("Error getting file size for %s: %s", entry_path, stat_err)
                     # Decide whether to proceed without size check or skip
                     return [], [] # Skip for safety

                mime_type = await self._get_mime_type(entry_path)
                file_ext = os.path.splitext(entry_path)[1].lower()

                if mime_type == 'application/zip' or file_ext == '.zip':
                    logger.info("Found zip file: %s. Extracting...", entry_path)
                    extracted_paths = await self._unzip_file(entry_path, os.path.dirname(entry_path), current_depth)
                    # Process the extracted files
                    tasks = [
                        self.process_entry(
                            extracted_path, current_depth + 1 # Depth increases for extracted items
                        ) for extracted_path in extracted_paths
                    ]
                    results = await asyncio.gather(*tasks)
                    for texts, images in results:
                        all_texts.extend(texts)
                        all_images.extend(images)

                # Check if file type is supported for conversion or direct reading
                elif (mime_type and any(mime_type.startswith(p) for p in SUPPORTED_TEXT_MIME_PREFIXES)) or \
                     (mime_type in SUPPORTED_TEXT_MIME_TYPES) or \
                     (mime_type in CONVERTIBLE_MIME_TYPES) or \
                     (file_ext in CONVERTIBLE_EXTENSIONS) or \
                     (mime_type and any(mime_type.startswith(p) for p in IMAGE_MIME_PREFIXES)):

                    texts, images = await convert_file_content(entry_path, mime_type or "")
                    all_texts.extend(texts)
                    all_images.extend(images)
                else:
                    logger.warning("Skipping unsupported file type: %s (MIME: %s)", entry_path, mime_type)

            else:
                 logger.warning("Skipping entry (not a file or directory, or access error): %s", entry_path)

        except Exception as e:
            logger.exception("Error processing entry %s: %s", entry_path, e)

        return all_texts, all_images


    async def get_student_content(self, student_dir: str) -> Tuple[List[str], List[str]]:
        """
        Scans a student's directory asynchronously and extracts all relevant content.

        Args:
            student_dir: The path to the student's submission directory.

        Returns:
            A tuple containing (list of all extracted text content, list of all base64 images).
        """
        logger.info("Starting content extraction for directory: %s", student_dir)
        if not os.path.isdir(student_dir): # Initial sync check is acceptable
             logger.error("Student directory not found or is not a directory: %s", student_dir)
             return [], []

        # Start processing from the root of the student directory at depth 0
        all_texts, all_images = await self.process_entry(os.path.abspath(student_dir), current_depth=0)

        logger.info("Finished content extraction for %s. Found %d text blocks, %d images.",
                    student_dir, len(all_texts), len(all_images))
        return all_texts, all_images