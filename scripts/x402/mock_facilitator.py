#!/usr/bin/env python3
"""Mock x402 facilitator for local MaskaraPay end-to-end testing.

This is a stdlib-only stand-in for a real facilitator (x402.org, Coinbase CDP,
...). It implements just enough of the facilitator HTTP API for MaskaraPay:

    GET  /supported  -> {"kind":"facilitator","schemes":[...]}   (discovery)
    POST /verify     -> {"isValid":bool,"invalidReason":str,"payer":str}
    POST /settle     -> {"success":bool,"errorReason":str, "transaction":str,
                         "network":str,"payer":str}

Success/failure behaves like a facilitator that "just works": /verify reports
the payer from the submitted payment payload, /settle mints a fake but
well-formed transaction hash. Two env switches make failure paths testable
without touching the server:

    MOCK_VERIFY_FAIL=1  /verify  reports isValid=false, invalidReason="insufficient_funds"
    MOCK_SETTLE_FAIL=1  /settle  reports success=false, errorReason="transaction_failed"

Run:

    python3 scripts/x402/mock_facilitator.py            # 127.0.0.1:8899
    python3 scripts/x402/mock_facilitator.py --port 9911

The mock is permissive on purpose: it does not validate signatures, amounts,
or networks. It is for local development only and must never be exposed
publicly or used with real funds.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Failure switches (read once at startup, simple and predictable).
MOCK_VERIFY_FAIL = os.environ.get("MOCK_VERIFY_FAIL", "").strip() == "1"
MOCK_SETTLE_FAIL = os.environ.get("MOCK_SETTLE_FAIL", "").strip() == "1"

# Advertised schemes. Both protocol spellings are listed (v1 uses friendly
# network names, v2 uses CAIP-2 ids) so discovery works for either version.
SUPPORTED = {
    "kind": "facilitator",
    "schemes": [
        {"scheme": "exact", "network": "base-sepolia"},
        {"scheme": "exact", "network": "eip155:84532"},
        {"scheme": "exact", "network": "base"},
        {"scheme": "exact", "network": "eip155:8453"},
    ],
}


def dig(data, *keys):
    """Walk a nested dict/list path defensively; None anywhere on the way."""
    node = data
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


class Handler(BaseHTTPRequestHandler):
    server_version = "MockX402Facilitator/1.0"

    # --- logging: "<METHOD> <path> <status>" on stderr -----------------------
    def log_request(self, code="-", size="-"):
        sys.stderr.write(f"{self.command} {self.path} {code}\n")
        sys.stderr.flush()

    def log_error(self, fmt, *args):  # keep errors in the same compact style
        sys.stderr.write(f"{self.command} {self.path} {fmt % args}\n")
        sys.stderr.flush()

    # --- helpers -------------------------------------------------------------
    def send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return body if isinstance(body, dict) else None

    # --- routes --------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/supported":
            self.send_json(200, SUPPORTED)
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        body = self.read_body_json()
        if body is None:
            self.send_json(400, {"error": "invalid_json"})
            return

        payment_payload = body.get("paymentPayload") or {}
        requirements = body.get("paymentRequirements") or {}
        payer = dig(payment_payload, "payload", "authorization", "from") or ""
        network = requirements.get("network") if isinstance(requirements, dict) else None
        network = network or ""

        if path == "/verify":
            if MOCK_VERIFY_FAIL:
                self.send_json(
                    200,
                    {"isValid": False, "invalidReason": "insufficient_funds", "payer": payer},
                )
            else:
                self.send_json(200, {"isValid": True, "invalidReason": "", "payer": payer})
        elif path == "/settle":
            if MOCK_SETTLE_FAIL:
                self.send_json(
                    200,
                    {
                        "success": False,
                        "errorReason": "transaction_failed",
                        "transaction": "",
                        "network": network,
                        "payer": payer,
                    },
                )
            else:
                self.send_json(
                    200,
                    {
                        "success": True,
                        "errorReason": "",
                        # 32 bytes of mock tx-hash, same shape as a real receipt.
                        "transaction": "0x" + secrets.token_hex(32),
                        "network": network,
                        "payer": payer,
                    },
                )
        else:
            self.send_json(404, {"error": "not_found"})


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Mock x402 facilitator for local MaskaraPay testing (stdlib only)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8899, help="bind port (default 8899)")
    args = parser.parse_args(argv)

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"error: cannot bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    httpd.daemon_threads = True

    print(f"mock x402 facilitator on http://{args.host}:{args.port}", flush=True)
    if MOCK_VERIFY_FAIL:
        print("  MOCK_VERIFY_FAIL=1: /verify will report insufficient_funds", flush=True)
    if MOCK_SETTLE_FAIL:
        print("  MOCK_SETTLE_FAIL=1: /settle will report transaction_failed", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass  # graceful Ctrl+C: stop accepting, drop worker threads, exit 0
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
