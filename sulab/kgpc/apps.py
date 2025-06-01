from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class KgpcConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'kgpc'
    verbose_name = 'Patent API'

    def ready(self):
        logger.info("Initializing KGPC application...")
        try:
            from .utils import model_loader
            model_loader.load_resources()
            logger.info("Resources loaded successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)