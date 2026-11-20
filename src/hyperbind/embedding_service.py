"""Loopback-only NV embedding service, run in the separate embed environment."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time

from .nv_adapter import NVEncoder


def serve(root: Path, port: int, batch_size: int, provider: str = 'nv'):
    import numpy as np
    if provider == 'nv':
        encoder = NVEncoder(root, batch_size=batch_size)
    elif provider == 'qwen4':
        from .qwen_embedding import QwenEncoder
        encoder = QwenEncoder(root, batch_size=batch_size)
    else:
        raise ValueError('provider must be nv or qwen4')

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'status': 'ready', 'settings': encoder.settings})
            else:
                self.reply(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/encode':
                self.reply(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8_000_000:
                    raise ValueError('request size limit')
                request = json.loads(self.rfile.read(length))
                texts = request['texts']
                if not isinstance(texts, list) or not 1 <= len(texts) <= 32:
                    raise ValueError('request must contain 1 to 32 texts')
                started = time.time()
                vectors = encoder.encode(texts, request.get('kind', 'query'),
                                         instruction=request.get('instruction')).astype('<f2')
                self.reply(200, {'dtype': 'float16', 'shape': list(vectors.shape),
                                 'data': base64.b64encode(vectors.tobytes()).decode(),
                                 'seconds': time.time() - started})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {'error': str(error)})
            except Exception as error:
                # No request text or credentials are logged.
                self.reply(500, {'error': type(error).__name__})
                raise

        def log_message(self, format, *args):
            print(format % args, flush=True)

    server = HTTPServer(('127.0.0.1', port), Handler)
    print(json.dumps({'status': 'ready', 'host': '127.0.0.1', 'port': port, 'settings': encoder.settings}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/hg_reasoning_agent'))
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--provider', choices=['nv', 'qwen4'], default='nv')
    args = parser.parse_args()
    serve(args.root, args.port, args.batch_size, args.provider)
