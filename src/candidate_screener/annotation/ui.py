"""A local labelling UI for the in-domain set — three screens, no new dependencies.

    uv run python -m candidate_screener.annotation.ui --annotator alice

Then open http://127.0.0.1:8765. Screens: the batch list, the new-batch form, and the
labelling screen (one JD, its candidates, three buttons each).

**Why the standard library and not Streamlit.** Two reasons, and the second is the one
that decided it. Adding a UI framework to `pyproject.toml` puts it in the dependency
closure of `uv sync` for everyone, including whoever only wants to re-run the baseline —
a cost paid by every clone for a tool two people use for one afternoon. More concretely,
creating a batch replays the whole campaign (~6 s: 141,897 JDs and 210,250 CVs loaded,
then a TF-IDF fit per JD), and a framework that re-executes the script on every widget
interaction is the wrong shape for an action like that. Here it is one POST.

**Both corpora are labelled here** *(D32, reversing D30)*. In-domain groups are 10
candidates; A1 recheck groups are however many that JD drew, 1 to 4. The two are
interleaved in each annotator's own order. `session.py` records what that costs and what
it does not: no `a1_label` and no `selection_reason` reach the page, so which pairs carry
an existing label — and what it says — is still not visible.

Everything the server does with data is in `session.py`. This file is transport: parse a
request, call a function, serialise the result. It holds no rule about what an annotator
may see, so there is nothing here for a test to have to reach through a socket to check.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from candidate_screener.annotation import queue as judging
from candidate_screener.annotation import sample, session

PAGE = Path(__file__).with_name("ui.html")

#: A new batch is capped well below the campaign's own limits. The cap is a guard against
#: a mistyped `n_jds` in a text box committing someone to thousands of judgements: the
#: whole session is budgeted at ~250 (D25), and a batch is frozen once appended.
MAX_NEW_JDS = 200


@lru_cache(maxsize=1)
def corpus_text() -> tuple[pd.Series, pd.Series]:
    """Query and document text for **both** corpora, loaded once per process (~3 s).

    A1 ids (`j_…`, `r_…`) and Djinni ids (UUIDs) cannot collide, so one lookup each is
    enough and the caller never has to know which corpus a group came from.
    """
    jd, cv = sample.load_english()
    queries = [jd.set_index("id").jd_text]
    documents = [cv.set_index("id").cv_text]
    try:
        recheck = judging.a1_recheck(judging.RECHECK_STRATA, 0)
        queries.append(recheck.set_index("query_id").query_text.groupby(level=0).first())
        documents.append(
            recheck.set_index("doc_id").candidate_text.groupby(level=0).first())
    except (FileNotFoundError, OSError):
        pass                      # in-domain still labellable without the A1 split
    return pd.concat(queries), pd.concat(documents)


@lru_cache(maxsize=1)
def units() -> pd.DataFrame:
    return session.load_units()


@lru_cache(maxsize=1)
def titles() -> list[dict]:
    return sample.available_titles().to_dict("records")


def next_group(annotator: str, batch: int | None) -> dict | None:
    jd_text, cv_text = corpus_text()
    group = session.serve_group(
        units(), session.load_judgements(), annotator, jd_text, cv_text, batch=batch)
    return group.as_dict() if group else None


def create_batch(selected: list[str], n_jds: int, per_jd: int = sample.PER_JD,
                 same_keyword_min: int | None = None) -> dict:
    """Append a batch. An empty or complete title selection means `generic` (D14).

    Selecting every title is not the same request as selecting none, but it is the same
    *batch*: both mean "do not scope this draw by role family", which is exactly what
    `keywords: null` records. Writing out all 41 titles instead would stamp `targeted` on
    the pairs and put them in the wrong reporting stratum for the rest of the project.
    """
    if not 1 <= n_jds <= MAX_NEW_JDS:
        raise ValueError(f"n_jds must be between 1 and {MAX_NEW_JDS}, got {n_jds}")
    if not 1 <= per_jd <= 20:
        raise ValueError(f"per_jd must be between 1 and 20, got {per_jd}")

    known = [t["title"] for t in titles()]
    keywords = None if not selected or set(selected) >= set(known) else sorted(selected)
    sample.validate_keywords(keywords, known)
    report = sample.add_batch(n_jds=n_jds, per_jd=per_jd, seed=None,
                              keywords=keywords, reuse_jds=False,
                              same_keyword_min=same_keyword_min)
    units.cache_clear()        # the manifest just grew; the cached frame is behind it
    return report["summary"]


class Handler(BaseHTTPRequestHandler):
    annotator = ""

    def log_message(self, *args) -> None:      # noqa: D102 - quiet by default
        pass

    def _send(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        route = urlparse(self.path)
        query = parse_qs(route.query)
        who = (query.get("annotator", [self.annotator])[0] or "").strip()
        try:
            if route.path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif route.path == "/api/state":
                self._send({"overview": session.batch_overview(who),
                            "titles": titles(),
                            "per_jd": sample.PER_JD,
                            "same_keyword_min": sample.SAME_KEYWORD_MIN,
                            "labels": list(session.LABELS),
                            "default_annotator": self.annotator})
            elif route.path == "/api/next":
                batch = query.get("batch", [""])[0]
                self._send({"group": next_group(who, int(batch) if batch else None)})
            else:
                self._send({"error": "not found"}, 404)
        except Exception as exc:               # surfaced in the page, not the console
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 400)

    def do_POST(self) -> None:
        route = urlparse(self.path)
        try:
            payload = self._body()
            if route.path == "/api/judge":
                # `corpus` and `selection_reason` are looked up from the manifest, never
                # taken from the request: they decide how the row is filed, and a page
                # that could set them could file an A1 recheck as an in-domain pair.
                rows = units()
                mine = rows[rows.jd_id.astype(str) == str(payload["jd_id"])]
                if mine.empty:
                    raise ValueError(f"unknown query {payload['jd_id']!r}")
                group = session.Group(
                    jd_id=payload["jd_id"], batch=int(mine.batch.iloc[0]),
                    stratum=str(mine.stratum.iloc[0]), jd_text="",
                    corpus=str(mine.corpus.iloc[0]),
                    selection_reason=str(mine.selection_reason.iloc[0]),
                    complete=bool(payload.get("complete", True)),
                    candidates=[session.Candidate(c, "") for c in payload["labels"]])
                written = session.record(
                    (payload.get("annotator") or self.annotator).strip(), group,
                    payload["labels"], payload.get("shortlist_pick", session.NO_PICK),
                    payload.get("notes", ""))
                self._send({"written": written})
            elif route.path == "/api/batch":
                self._send({"summary": create_batch(
                    payload.get("titles") or [], int(payload.get("n_jds", 20)),
                    int(payload.get("per_jd", sample.PER_JD)),
                    int(payload.get("same_keyword_min", sample.SAME_KEYWORD_MIN)))})
            else:
                self._send({"error": "not found"}, 404)
        except Exception as exc:
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 400)


def serve(annotator: str, port: int, open_browser: bool = True) -> None:
    Handler.annotator = annotator
    corpus_text()                              # pay the load before the first request
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"\n=== Annotation UI — {url}  (annotator: {annotator or 'set in the page'})")
    print("  in-domain pairs only; the A1 recheck runs from the dispatch CSV")
    print(f"  labels append to {session.JUDGEMENTS}")
    print("  bound to 127.0.0.1 — these are real CVs, do not expose this port")
    print("  Ctrl-C to stop\n")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. Re-run to resume — progress lives in judgements.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", default="",
                    help="who is labelling — stamped on every row")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    serve(args.annotator, args.port, not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
