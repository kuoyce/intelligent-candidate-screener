"""Local, thin HTTP transport for human entity annotation."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

HTML = Path(__file__).with_name("ui.html")
MAX_REQUEST_BYTES = 2_000_000


def create_server(service, host="127.0.0.1", port=8766):
    if host != "127.0.0.1":
        raise ValueError("Entity annotation must bind to 127.0.0.1")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # No document IDs, annotator IDs or request bodies in access logs.

        def reply(self, status, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def error(self, exc):
            status = getattr(exc, "status_code", None)
            if status is None:
                status = 409 if type(exc).__name__ in {"RevisionConflict", "ConflictError"} else 400
            if status not in {400, 403, 404, 409, 422}:
                status = 400
            self.reply(status, {"error": str(exc) if isinstance(exc, (ValueError, KeyError)) or status == 409 else "Request failed"})

        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                self.reply(403, {"error": "Local host only"})
                return
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self.reply(200, HTML.read_bytes(), "text/html; charset=utf-8")
                elif parsed.path == "/api/session":
                    self.reply(200, service.session(params.get("annotator", [""])[0]))
                elif parsed.path == "/api/document":
                    self.reply(200, service.document(params.get("doc_id", [""])[0], params.get("annotator", [""])[0]))
                elif parsed.path == "/api/suggestions":
                    self.reply(200, service.suggestions(params.get("doc_id", [""])[0], params.get("annotator", [""])[0]))
                elif parsed.path == "/boundary-review":
                    page = service.boundary_review(params.get("annotator", [""])[0])
                    self.reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
                else:
                    self.reply(404, {"error": "Unknown route"})
            except Exception as exc:
                self.error(exc)

        def do_POST(self):
            # Reject cross-origin form/JSON writes, including DNS rebinding hosts.
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected or self.headers.get("Origin", f"http://{expected}") != f"http://{expected}":
                self.reply(403, {"error": "Local same-origin requests only"})
                return
            if urlsplit(self.path).path != "/api/save":
                self.reply(404, {"error": "Unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request size")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected JSON request")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected JSON object")
                self.reply(200, service.save(payload))
            except Exception as exc:
                self.error(exc)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--annotator", default="")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if bool(args.campaign) == bool(args.demo):
        parser.error("Choose exactly one of --campaign or --demo")
    if args.demo:
        from .demo import create_demo
        args.campaign = create_demo()
    from .service import AnnotationService
    server = create_server(AnnotationService(args.campaign), port=args.port)
    from urllib.parse import quote
    print(f"Entity annotation: http://127.0.0.1:{server.server_port}/?annotator={quote(args.annotator)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
