"""Quiet Streamlit's module-watcher warnings about transformers' lazy submodules.

Loading the reranker imports ``transformers``. Streamlit's watcher then probes every
module in ``sys.modules``, which trips transformers' lazy loader into importing
image processors that need ``torchvision`` (not installed, never used). Streamlit
catches the error and logs a traceback per submodule. This drops only those
records; every other watcher warning, and file watching itself, is unchanged.
"""
from __future__ import annotations

import logging

WATCHER_LOGGER = "streamlit.watcher.local_sources_watcher"
_NOISY_PREFIX = "transformers."


class TransformersLazyImportFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not str(record.msg).startswith("Examining the path of"):
            return True
        name = str(record.args[0]) if record.args else ""
        return not name.startswith(_NOISY_PREFIX)


def install() -> None:
    """Attach the filter once; Streamlit re-executes the app script on every rerun."""
    logger = logging.getLogger(WATCHER_LOGGER)
    if not any(isinstance(f, TransformersLazyImportFilter) for f in logger.filters):
        logger.addFilter(TransformersLazyImportFilter())
