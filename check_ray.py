import ray
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"Ray version: {ray.__version__}")

try:
    logger.info("Attempting to initialize Ray...")
    ray.init(ignore_reinit_error=True, logging_level=logging.DEBUG)
    logger.info("Ray initialized successfully")
    logger.info(f"Ray resources: {ray.available_resources()}")
except Exception as e:
    logger.exception(f"Failed to initialize Ray: {str(e)}")
    sys.exit(1)

logger.info("Ray initialization check completed")
