import logging

import pytest

from candidate_screener.app import watcher_logging


@pytest.fixture
def watcher_logger():
    logger = logging.getLogger(watcher_logging.WATCHER_LOGGER)
    saved = list(logger.filters)
    logger.filters.clear()
    yield logger
    logger.filters[:] = saved


def _messages(logger, caplog, name):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=logger.name):
        logger.warning("Examining the path of %s raised:", name)
    return [r.getMessage() for r in caplog.records]


def test_drops_transformers_lazy_import_tracebacks(watcher_logger, caplog):
    watcher_logging.install()
    assert _messages(watcher_logger, caplog, "transformers.models.yolos.image_processing_yolos_fast") == []


def test_keeps_other_watcher_warnings(watcher_logger, caplog):
    watcher_logging.install()
    assert _messages(watcher_logger, caplog, "candidate_screener.app.portal_ui")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=watcher_logger.name):
        watcher_logger.warning("Failed to watch file %s", "transformers.py")
    assert len(caplog.records) == 1


def test_install_is_idempotent_across_reruns(watcher_logger):
    watcher_logging.install()
    watcher_logging.install()
    assert len(watcher_logger.filters) == 1
