"""Rank-aware logging with Rich console output and optional file logging.

All log methods are decorated with ``@zero_rank`` so that only the main
process (rank 0) emits messages in distributed training.

Usage::

    from unimate.utils.logger import get_logger
    logger = get_logger(file_name=__file__)
    logger.info("Training started")
"""

import inspect
import logging
import os
from typing import Optional

import torch.distributed as dist
from rich.logging import RichHandler


# -------------------------------------------------------------------
# Distributed helper
# -------------------------------------------------------------------

def zero_rank(func):
    """Decorator that silences the wrapped method on non-zero ranks."""
    def wrapper(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper


# -------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------

class Logger:
    """Thin wrapper around :mod:`logging` with Rich formatting and rank gating.

    Args:
        file_name: Name used for the underlying ``logging.Logger`` and the
            log file (when *log_dir* is provided).
        log_dir: If set, a ``FileHandler`` is added that writes to
            ``<log_dir>/<file_name>.log``.
        level: ``"info"`` or ``"debug"`` (case-insensitive).
    """

    def __init__(self, file_name: str = "log", log_dir: str = None,
                 level: str = "info"):
        self.logger = logging.getLogger(file_name)
        self.logger.propagate = False

        level = self._resolve_level(level)
        self.level = level
        self.logger.setLevel(level)

        # Clear stale handlers (e.g. after re-instantiation)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler (Rich)
        console_handler = RichHandler(rich_tracebacks=True, show_path=False)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            self._CallerFileFormatter("%(message)s (%(filename)s)")
        )
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_dir:
            log_file = os.path.join(log_dir, f"{file_name}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(
                self._CallerFileFormatter(
                    "%(asctime)s - %(levelname)s - %(message)s (%(filename)s)"
                )
            )
            self.logger.addHandler(file_handler)

    # ---------------------------------------------------------------
    # Log methods (only rank 0 in distributed training)
    # ---------------------------------------------------------------

    @zero_rank
    def debug(self, message):
        self.logger.debug(message)

    @zero_rank
    def info(self, message):
        self.logger.info(message)

    @zero_rank
    def warning(self, message):
        self.logger.warning(message)

    @zero_rank
    def error(self, message):
        self.logger.error(message)

    @zero_rank
    def critical(self, message):
        self.logger.critical(message)

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _resolve_level(level: str) -> int:
        """Convert a level string to a ``logging`` constant."""
        level = level.upper()
        if level == "INFO":
            return logging.INFO
        if level == "DEBUG":
            return logging.DEBUG
        raise ValueError(f"Invalid log level: {level}")

    class _CallerFileFormatter(logging.Formatter):
        """Formatter that resolves ``%(filename)s`` to the *actual* caller,
        skipping internal logging / Rich frames."""

        def format(self, record):
            frame = inspect.currentframe()
            while frame:
                co = frame.f_code.co_filename
                if (co != __file__
                        and "logging" not in co
                        and "rich" not in co):
                    record.filename = os.path.basename(co)
                    break
                frame = frame.f_back
            if frame is None:
                record.filename = "unknown"
            return super().format(record)


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------

def get_logger(file_name: Optional[str] = None,
               debug: Optional[str] = "", **kwargs) -> Logger:
    """Create a :class:`Logger` instance.

    The log level is set to ``DEBUG`` when the ``DEBUG`` env var matches
    any value in *debug*; otherwise defaults to ``INFO``.

    Args:
        file_name: Passed through to :class:`Logger`.
        debug: A string (or list of strings) compared against ``$DEBUG``.
        **kwargs: Forwarded to :class:`Logger` (e.g. ``log_dir``).
    """
    if isinstance(debug, str):
        debug = [debug]
    level = "DEBUG" if os.environ.get("DEBUG") in debug else "INFO"
    return Logger(file_name=file_name, level=level, **kwargs)
