import django
import logging
from django.conf import settings
from django.core.exceptions import AppRegistryNotReady
from human_feedback_site import settings as site_settings
__version__ = "0.1.0"


def initialize():
    try:
        valid_settings = {key: value for key,
                          value in site_settings.__dict__.items() if key.isupper()}
        settings.configure(**valid_settings)
        django.setup()
    except RuntimeError:
        logging.warning(
            "Tried to double configure the API, ignore this if running the Django app directly")


initialize()

try:
    from human_feedback_api.models import Comparison
except AppRegistryNotReady:
    logging.info("Could not yet import Feedback")
