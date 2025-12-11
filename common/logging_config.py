from __future__ import annotations

import logging
from logging.config import dictConfig


def configure_logging(level: str = "INFO", service_name: str = "app") -> None:
    """Configure structured logging for the application.

    This sets up a simple console logger with timestamps, log level, and
    service name. It is intentionally minimal but suitable for container
    deployments.
    """

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": (
                        "%(asctime)s | %(levelname)s | %(name)s | "
                        + service_name
                        + " | %(message)s"
                    ),
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": level.upper(),
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level.upper(),
            },
        }
    )

    logging.getLogger(__name__).info("Logging configured for service '%s'", service_name)
