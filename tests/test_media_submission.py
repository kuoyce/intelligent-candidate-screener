import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from candidate_screener.multimodal.submission import MediaError, PREFIX, process_media, validate_upload


def response():
    return {"source_language":"fr", "status":"translated", "segments":[
        {"start":0,"end":3,"french":"bonjour", "english":"hello"}]}


def test_upload_uses_private_name_and_deletes_bytes_after_success():
    paths=[]
    def runner(command, **kwargs):
        path=Path(command[command.index("--media")+1]);paths.append(path)
        assert path.name == "upload.mp4" and path.read_bytes() == b"media"
        assert kwargs["timeout"] == 240 and kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        return SimpleNamespace(returncode=0,stdout=PREFIX+json.dumps(response()),stderr="")
    result=process_media("private-name.mp4",b"media",runner=runner,command=["worker"])
    assert result["review_required"]
    assert not paths[0].exists()


def test_timeout_deletes_upload_and_hides_sensitive_exception():
    paths=[]
    def runner(command, **kwargs):
        paths.append(Path(command[command.index("--media")+1]))
        raise subprocess.TimeoutExpired("private-name",240,stderr="private transcript")
    with pytest.raises(MediaError,match="shorter") as error:
        process_media("name.mp4",b"media",runner=runner,command=["worker"])
    assert "private" not in str(error.value)
    assert not paths[0].exists()


@pytest.mark.parametrize("name,content,language",[("cv.exe",b"x","fr"),("cv.mp4",b"","fr"),
                                                  ("cv.mp4",b"x","de")])
def test_invalid_upload_never_starts_worker(name,content,language):
    with pytest.raises(MediaError):
        process_media(name,content,language,runner=lambda *a,**k: pytest.fail("worker started"))


def test_worker_language_mismatch_cannot_be_confirmed():
    data=response();data["source_language"]="en"
    with pytest.raises(MediaError,match="complete"):
        process_media("cv.mp4",b"x",runner=lambda *a,**k: SimpleNamespace(
            returncode=0,stdout=PREFIX+json.dumps(data)),command=["worker"])


def test_negative_or_overlong_timestamp_rejected():
    data=response();data["segments"][0]["end"]=121
    with pytest.raises(MediaError,match="incomplete"):
        process_media("cv.mp4",b"x",runner=lambda *a,**k: SimpleNamespace(
            returncode=0,stdout=PREFIX+json.dumps(data)),command=["worker"])
