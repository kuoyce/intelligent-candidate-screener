"""Loopback-only frozen Qwen inference server for the isolated transformer runtime.

This module deliberately imports only the standard library at module scope: the
isolated runtime need not install the main package's pandas/Streamlit stack.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import resource
import sys
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock
from candidate_screener.llm.qwen_assets import asset_digest
from candidate_screener.llm.local_reranker import load_rerank_runtime

MAX_BODY_BYTES = 1024 * 1024
MAX_RERANK_BODY_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_TOKENS = 256
MAX_CONTEXT_TOKENS = 4096
MAX_GENERATION_SECONDS = 60


def checkpoint_digest(checkpoint: Path) -> str:
    """Hash relative names and bytes of every checkpoint file in sorted order."""
    return asset_digest(checkpoint)


def validate_messages(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 16:
        raise ValueError('Invalid messages')
    clean = []
    for message in value:
        if not isinstance(message, dict) or set(message) != {'role', 'content'}:
            raise ValueError('Invalid message fields')
        if not isinstance(message['role'], str) or message['role'] not in {'system', 'user', 'assistant'} or not isinstance(message['content'], str):
            raise ValueError('Invalid message role or content')
        if len(message['content']) > 100000:
            raise ValueError('Message too large')
        clean.append(dict(message))
    return clean


class LocalQwenRuntime:
    """Single-process inference facade; documents never enter a persistent cache."""
    def __init__(self, tokenizer, model, torch, *, model_id, revision, sha256, versions=None):
        self.tokenizer, self.model, self.torch = tokenizer, model, torch
        self.model_id, self.revision, self.sha256 = model_id, revision, sha256
        self.versions = versions or {}
        self.lock = Lock()

    def token_count(self, messages):
        messages = validate_messages(messages)
        with self.lock:
            return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def complete(self, messages, max_tokens):
        messages = validate_messages(messages)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError('Invalid output token limit')
        with self.lock:
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        return_tensors='pt', return_dict=True)
            prompt_tokens = inputs['input_ids'].shape[-1]
            if prompt_tokens + max_tokens > MAX_CONTEXT_TOKENS:
                raise ValueError('Context budget exceeded')
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens,
                                             max_time=MAX_GENERATION_SECONDS, use_cache=True,
                                             pad_token_id=self.tokenizer.eos_token_id,
                                             temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)
            generated = output[0][prompt_tokens:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            eos = getattr(getattr(self.model, 'generation_config', None), 'eos_token_id', self.tokenizer.eos_token_id)
            stops = eos if isinstance(eos, (list, tuple)) else [eos]
            reason = 'stop' if len(generated) and generated[-1] in stops else 'length'
            return {'content': text, 'prompt_tokens': prompt_tokens, 'completion_tokens': len(generated), 'finish_reason': reason}

    def metadata(self):
        return {'id': self.model_id, 'object': 'model', 'revision': self.revision, 'sha256': self.sha256,
                'runtime': self.versions, 'max_context_tokens': MAX_CONTEXT_TOKENS,
                'max_output_tokens': MAX_OUTPUT_TOKENS, 'device': 'cpu', 'dtype': 'float32',
                'generation': {'do_sample': False, 'repetition_penalty': 1.0, 'max_time_seconds': MAX_GENERATION_SECONDS},
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}


def load_runtime(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or any(not isinstance(manifest.get(k), str) or not manifest[k]
                                              for k in ('model', 'revision', 'checkpoint', 'sha256')):
        raise ValueError('Invalid frozen model manifest')
    if not re.fullmatch(r'[0-9a-f]{40}', manifest['revision']):
        raise ValueError('Model revision must be a pinned commit')
    checkpoint = Path(manifest['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = manifest_path.parent / checkpoint
    checkpoint = checkpoint.resolve()
    if checkpoint_digest(checkpoint) != manifest['sha256']:
        raise ValueError('Checkpoint digest mismatch')
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False,
                                               torch_dtype=torch.float32)
    model.to('cpu')
    model.eval()
    return LocalQwenRuntime(tokenizer, model, torch, model_id=manifest['model'], revision=manifest['revision'],
                            sha256=manifest['sha256'], versions={'torch': torch.__version__, 'transformers': transformers.__version__})


def make_handler(runtime, reranker=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # HTTP request paths/bodies and prompts are never operational logs.
            pass

        def send_json(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/v1/models':
                self.send_json(200, {'object': 'list', 'data': [runtime.metadata()]})
            else:
                self.send_json(404, {'error': {'message': 'Unknown endpoint'}})

        def do_POST(self):
            if self.path not in {'/v1/chat/completions', '/v1/tokenize', '/v1/rerank'}:
                self.send_json(404, {'error': {'message': 'Unknown endpoint'}})
                return
            try:
                raw_length = self.headers.get('Content-Length', '')
                limit = MAX_RERANK_BODY_BYTES if self.path == '/v1/rerank' else MAX_BODY_BYTES
                if not raw_length.isdigit() or not 0 < int(raw_length) <= limit:
                    raise ValueError('Invalid body length')
                self.connection.settimeout(10)
                body = json.loads(self.rfile.read(int(raw_length)))
                if self.path == '/v1/rerank':
                    if reranker is None:
                        self.send_json(503, {'error': {'message': 'Frozen reranker is not provisioned'}})
                        return
                    self.send_json(200, reranker.score(body))
                    return
                if not isinstance(body, dict) or body.get('model') != runtime.model_id:
                    raise ValueError('Invalid model')
                allowed = {'model', 'messages'} if self.path == '/v1/tokenize' else {'model', 'messages', 'max_tokens', 'temperature', 'stream', 'top_p'}
                if not set(body).issubset(allowed):
                    raise ValueError('Unsupported request fields')
                messages = validate_messages(body.get('messages'))
                if self.path == '/v1/tokenize':
                    self.send_json(200, {'count': runtime.token_count(messages)})
                    return
                if body.get('stream', False) is not False or body.get('temperature', 0) != 0:
                    raise ValueError('Only greedy nonstreaming inference supported')
                top_p = body.get('top_p', 1)
                if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or top_p != 1:
                    raise ValueError('Only default top_p supported')
                answer = runtime.complete(messages, body.get('max_tokens', MAX_OUTPUT_TOKENS))
                self.send_json(200, {'object': 'chat.completion', 'model': runtime.model_id,
                                     'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer['content']}, 'finish_reason': answer.get('finish_reason', 'stop')}],
                                     'usage': {'prompt_tokens': answer['prompt_tokens'], 'completion_tokens': answer['completion_tokens'],
                                               'total_tokens': answer['prompt_tokens'] + answer['completion_tokens']}})
            except (ValueError, TypeError, KeyError, TimeoutError, json.JSONDecodeError):
                self.send_json(400, {'error': {'message': 'Invalid or unsupported inference request'}})
            except Exception:
                self.send_json(503, {'error': {'message': 'Local inference unavailable'}})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--reranker-manifest', help='Optional frozen CE manifest; otherwise discover the local exported manifest')
    parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1'])
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)
    started = time.monotonic()
    runtime = load_runtime(args.manifest)
    rerank_path = Path(args.reranker_manifest) if args.reranker_manifest else Path(args.manifest).resolve().parent.parent / 'reranker.json'
    reranker = load_rerank_runtime(rerank_path) if rerank_path.is_file() else None
    if args.reranker_manifest and reranker is None:
        parser.error('Configured reranker manifest is unavailable')
    with HTTPServer((args.host, args.port), make_handler(runtime, reranker)) as server:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(json.dumps({'status': 'ready', 'host': args.host, 'port': args.port, 'model': runtime.metadata(),
                          'load_seconds': time.monotonic() - started,
                          'peak_rss_bytes': rss if sys.platform == 'darwin' else rss * 1024,
                          'reranker': reranker.metadata() if reranker else None}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
