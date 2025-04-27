# -*- coding: utf-8 -*-
"""Main script to run the Grading Agent."""

import asyncio
import logging
import os
import sys

# Ensure the grading_agent package is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from grading_agent.config.config_manager import ConfigManager, get_config_manager
from grading_agent.core.agent import GradingAgent


def setup_logging(config: ConfigManager):
    """Sets up logging based on the configuration."""
    log_level_str = config.get('logging.level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('logging.log_file') # Optional log file

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level) # Set root level

    # Clear existing handlers (important if re-running in same process)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level) # Use same level for console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a') # Append mode
            file_handler.setLevel(log_level) # Use same level for file
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info("Logging to file: %s", log_file)
        except Exception as e:
            logging.error("Failed to set up file logging to %s: %s", log_file, e, exc_info=True)

    logging.info("Logging setup complete. Level: %s", log_level_str)


async def main():
    """Initializes and runs the Grading Agent."""
    config = None
    try:
        # Load configuration first to setup logging based on it
        config = get_config_manager() # Uses default paths 'config.yaml', '.env'
        setup_logging(config)

        # Now create and run the agent
        agent = GradingAgent(config_manager=config)
        max_concurrency = config.get('agent.max_concurrent_students', 5)
        await agent.run(max_concurrent_students=max_concurrency)

    except FileNotFoundError as e:
        # Log file not found errors specifically (e.g., config.yaml)
        logging.error("Initialization failed: %s", e, exc_info=True)
        print(f"Error: Required file not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        # Log configuration errors
        logging.error("Configuration error: %s", e, exc_info=True)
        print(f"Error: Configuration issue - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during setup or run
        logging.critical("An unexpected critical error occurred: %s", e, exc_info=True)
        print(f"Critical Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Check Python version (AsyncIO features used might require >= 3.7 or higher)
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required to run this agent.", file=sys.stderr)
        sys.exit(1)

    # Run the main asynchronous function
    asyncio.run(main())