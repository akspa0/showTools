# test_imports.py
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Attempting to import lmstudio and introspect...")
try:
    import lmstudio as lms
    logger.info("Successfully imported lmstudio as lms.")
    logger.info("--- dir(lms) ---")
    print(dir(lms))
    logger.info("--- End dir(lms) ---")

    common_exceptions = [
        "APIConnectionError", 
        "LMStudioSDKError", 
        "APIError", 
        "AuthenticationError", 
        "NotFoundError", 
        "RateLimitError", 
        "UnprocessableEntityError",
        "InternalServerError",
        "Timeout",
        "InvalidRequestError"
    ]

    logger.info("--- Checking for common exceptions as attributes of lms ---")
    for exc_name in common_exceptions:
        if hasattr(lms, exc_name):
            logger.info(f"Found lms.{exc_name}")
        else:
            logger.info(f"lms.{exc_name} NOT found")
    logger.info("--- End exception check ---")

except ImportError as e:
    logger.error(f"Failed to import lmstudio: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred during lmstudio import or introspection: {e}")

logger.info("--- Attempting to import other project modules (after lmstudio test) ---")
try:
    import audio_preprocessor
    logger.info("Successfully imported audio_preprocessor")
    import clap_module
    logger.info("Successfully imported clap_module")
    import diarization_module
    logger.info("Successfully imported diarization_module")
    # We will still try to import llm_module to see if IT causes an error now,
    # even if lmstudio itself has issues with submodules.
    import llm_module 
    logger.info("Successfully imported llm_module")
    import transcription_module
    logger.info("Successfully imported transcription_module")
    logger.info("All main project modules imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import a module after lmstudio test: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred during other module imports: {e}")

logger.info("Script finished.") 