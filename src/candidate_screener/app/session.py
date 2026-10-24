"""Session invalidation without a dependency on Streamlit or model packages."""
from __future__ import annotations

import hashlib
import json

SCREENING_KEYS = ("snapshot", "results", "cvs", "chat", "chat_state", "last_answer", "rag_service", "filter_description")


def input_key(jd_text, jd_title, source, documents, configuration):
    """Document digests stay session-local; filenames never identify applicants."""
    content = [hashlib.sha256(d if isinstance(d, bytes) else str(d).encode()).hexdigest()
               for d in documents]
    value = {"jd": jd_text, "title": jd_title, "source": source,
             "documents": content, "config": configuration}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def invalidate(state, current_key):
    """Clears old results/history immediately, even before another screening action."""
    previous = state.get("screening_input_key")
    changed = previous is not None and previous != current_key
    if changed:
        service = state.get("rag_service")
        if hasattr(service, "close"):
            service.close()
        for key in SCREENING_KEYS:
            state.pop(key, None)
    state["screening_input_key"] = current_key
    return changed
