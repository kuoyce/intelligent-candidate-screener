"""The Streamlit transport starts, and uploaded files become text."""
from pathlib import Path

import pytest

from candidate_screener.app import files


def test_html_and_text_uploads_become_plain_text():
    html = b"<html><body><h1>Jane</h1><p>Python &amp; SQL</p><script>x()</script></body></html>"
    assert files.to_text("cv.html", html).split() == ["Jane", "Python", "&", "SQL"]
    assert files.to_text("cv.txt", b"  Python   developer ") == "Python developer"


def test_the_app_renders_its_empty_state():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    app = Path(__file__).resolve().parents[1] / "src/candidate_screener/app/streamlit_app.py"
    at = st_testing.AppTest.from_file(str(app), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Screen" in b.label for b in at.button)
