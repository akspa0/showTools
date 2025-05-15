import logging
import sys
from config import settings # Assuming settings.py is in ../config/

def setup_logging():
    """Configures logging for the application."""
    log_level = settings.LOG_LEVEL
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers if any (important for Gradio reload)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    # Optional: Add file handler
    # log_file = settings.PROJECT_ROOT / 'app.log'
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(log_level)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    logging.info(f"Logging configured with level: {log_level}")

# Configure logging when this module is imported
# setup_logging() 
# Consider calling setup_logging() explicitly in the main app entry point (e.g., gradio_app/app.py) 
# to ensure it runs at the right time, especially with tools like Gradio that might reload modules. 