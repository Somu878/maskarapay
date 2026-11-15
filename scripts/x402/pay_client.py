#!/usr/bin/env python3
"""x402 payment client for MaskaraPay.

Drives the whole HTTP-402 payment dance by hand (no x402 SDK):

    1. POST /api/v1/redact                       -> 402 Payment Required
    2. parse the payment requirements            (v2: PAYMENT-REQUIRED header,
                                                  v1: the 402 response body)
    3. sign an EIP-3009 USDC transferWithAuthorization (EIP-712)
    4. retry with the payment header             (v2: PAYMENT-SIGNATURE,
                                                  v1: X-PAYMENT)
    5. decode the settlement receipt             (v2: PAYMENT-RESPONSE,
                                                  v1: X-PAYMENT-RESPONSE)

Dependencies (see scripts/x402/requirements.txt): `requests` and
`eth-account`. The EIP-712 typed data is assembled directly (Domain +
TransferWithAuthorization), signed with eth_account, and the 65-byte
signature is hex-encoded.

Usage:

    python3 -m venv .venv && .venv/bin/pip install -r scripts/x402/requirements.txt
    export MASKARA_PRIVATE_KEY=0x...        # never printed by this script
    .venv/bin/python scripts/x402/pay_client.py http://127.0.0.1:8080 --version 2

Exit status: 0 if every requested protocol version completed payment +
settlement + inference, 1 otherwise.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import time

import requests
from eth_account import Account

# --- protocol constants ------------------------------------------------------

DEFAULT_TEXT = "Email ravi.sharma@example.com about the invoice."

# Header names, per protocol version.
HEADER_CHALLENGE_V2 = "PAYMENT-REQUIRED"     # server -> client: v2 402 challenge
HEADER_PAYMENT_V2 = "PAYMENT-SIGNATURE"      # client -> server: v2 payment
HEADER_RECEIPT_V2 = "PAYMENT-RESPONSE"       # server -> client: v2 settlement
HEADER_PAYMENT_V1 = "X-PAYMENT"              # client -> server: v1 payment
HEADER_RECEIPT_V1 = "X-PAYMENT-RESPONSE"     # server -> client: v1 settlement

# Per-network USDC constants. The EIP-712 domain name differs between the
# testnet token ("USDC") and Base mainnet ("USD Coin"); it must match the
# deployed asset or the facilitator / on-chain recovery rejects the signature.
NETWORKS = {
    "base-sepolia": {
        "chain_id": 84532,
        "caip2": "eip155:84532",
        "usdc": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
        "domain_name": "USDC",
        "domain_version": "2",
    },
    "base": {
        "chain_id": 8453,
        "caip2": "eip155:8453",
        "usdc": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        "domain_name": "USD Coin",
        "domain_version": "2",
    },
}

USDC_DECIMALS = 6  # human-readable display only; amounts come from the server
CLOCK_SKEW = 30    # allow the payer clock to be a bit ahead of the chain
REQUEST_TIMEOUT = 20.0


# --- small utilities ---------------------------------------------------------

def info(msg):
    print(f"[x402] {msg}", flush=True)


def encode_b64_json(obj):
    """Compact JSON -> standard base64 (padded), as the server expects."""
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def decode_b64_json(text):
    """base64 (or base64url) -> dict; raises ValueError when undecodable."""
    text = text.strip()
    pad = "=" * (-len(text) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            return json.loads(decoder(text + pad))
        except Exception:  # fall through and try the other alphabet
            continue
    raise ValueError("not base64 JSON")


def body_preview(resp, limit=300):
    try:
        return resp.text[:limit].replace("\n", " ")
    except Exception:
        return "<unreadable body>"


# --- challenge parsing -------------------------------------------------------

def load_challenge(resp, version):
    """Return the x402 challenge dict for `version`, or None if not offered.

    v2 challenges live in the base64 PAYMENT-REQUIRED response header,
    v1 challenges in the JSON body of the 402 itself.
    """
    if version == 2:
        raw = resp.headers.get(HEADER_CHALLENGE_V2)
        if not raw:
            return None
        try:
            challenge = decode_b64_json(raw)
        except ValueError:
            return None
    else:
        try:
            challenge = resp.json()
        except ValueError:
            return None
    if not isinstance(challenge, dict) or challenge.get("x402Version") != version:
        return None
    return challenge


def pick_requirement(accepts, network):
    """Pick the `exact` entry matching our network (name or CAIP-2 id)."""
    targets = {network, NETWORKS[network]["caip2"]}
    for req in accepts or []:
        if not isinstance(req, dict):
            continue
        if req.get("scheme") == "exact" and req.get("network") in targets:
            return req
    return None


# --- signing -----------------------------------------------------------------

def sign_transfer_with_authorization(account, requirement, value, valid_after, valid_before):
    """Sign a USDC EIP-3009 transferWithAuthorization (EIP-712).

    `value` is the amount as an int of USDC atomic units, taken verbatim from
    the payment requirements ("10000" == $0.01 at 6 decimals). Returns the
    65-byte signature as "0x"-hex along with the authorization object:
    a signed promise to let the USDC contract transfer `value` from us to
    the payee, valid for the given time window — the facilitator settles it
    on-chain later.

    Note the EIP-712 layout: value is uint256, validAfter/validBefore are
    uint48, nonce is a fresh random bytes32 (a new nonce = a new, non-replayable
    grant). The domain name/version/contract come from the per-network USDC
    constants, since the domain must match the deployed asset.
    """
    net = NETWORKS[_network_key(requirement)]
    nonce = secrets.token_bytes(32)
    typed_data = {
        "types": {
            "EIP712Domain": [
                {"name": "name", "type": "string"},
                {"name": "version", "type": "string"},
                {"name": "chainId", "type": "uint256"},
                {"name": "verifyingContract", "type": "address"},
            ],
            "TransferWithAuthorization": [
                {"name": "from", "type": "address"},
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "validAfter", "type": "uint48"},
                {"name": "validBefore", "type": "uint48"},
                {"name": "nonce", "type": "bytes32"},
            ],
        },
        "primaryType": "TransferWithAuthorization",
        "domain": {
            "name": net["domain_name"],
            "version": net["domain_version"],
            "chainId": net["chain_id"],
            "verifyingContract": net["usdc"],
        },
        "message": {
            "from": account.address,
            "to": requirement["payTo"],
            "value": int(value),
            "validAfter": int(valid_after),
            "validBefore": int(valid_before),
            "nonce": nonce,
        },
    }
    # eth_account renamed this entrypoint a few times; both spellings return a
    # signed message with a 65-byte .signature (r || s || v).
    signer = getattr(account, "unsafe_sign_typed_data", None) or account.sign_typed_data
    signed = signer(typed_data)
    signature = "0x" + signed.signature.hex()
    authorization = {
        "from": account.address,
        "to": requirement["payTo"],
        "value": str(int(value)),
        "validAfter": str(int(valid_after)),
        "validBefore": str(int(valid_before)),
        "nonce": "0x" + nonce.hex(),
    }
    return signature, authorization


def _network_key(requirement):
    """Map a requirement's network field to a key of NETWORKS."""
    net = (requirement.get("network") or "").lower()
    for key, cfg in NETWORKS.items():
        if net in (key, cfg["caip2"]):
            return key
    raise ValueError(f"unsupported network in requirements: {requirement.get('network')!r}")


def load_account(private_key_flag):
    """Build the payer account from --private-key or MASKARA_PRIVATE_KEY.

    The key material is never printed, logged, or included in error text.
    """
    key = private_key_flag or os.environ.get("MASKARA_PRIVATE_KEY") or ""
    key = key.strip()
    if not key:
        raise SystemExit(
            "[x402] no private key: set MASKARA_PRIVATE_KEY or pass --private-key\n"
            "       (0x + 64 hex chars; keep the key in the environment or a shell-"
            "ignored .env — never commit it)"
        )
    if not key.startswith("0x"):
        key = "0x" + key
    try:
        return Account.from_key(key)
    except Exception:
        # Deliberately generic: the underlying error could echo key material.
        raise SystemExit("[x402] invalid private key (expected 0x + 64 hex chars)")


# --- one full payment attempt for one protocol version -----------------------

def run_flow(args, version, account):
    server = args.server_url.rstrip("/")
    url = f"{server}/api/v1/redact"
    payload = {"model": "somukandula/maskara", "text": args.text}
    base_headers = {"content-type": "application/json"}
    session = requests.Session()

    print(f"--- x402 protocol version {version} ---", flush=True)

    # Step 1: ordinary request; expect either a free 200 (payments off) or a 402.
    try:
        resp = session.post(url, json=payload, headers=base_headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        info(f"request failed: {exc.__class__.__name__}: {exc}")
        return False
    info(f"POST {url} -> {resp.status_code} {resp.reason}")

    if resp.status_code == 405 or resp.status_code == 404:
        info(f"server rejected the endpoint; body: {body_preview(resp)}")
        return False
    if resp.status_code == 200:
        # No payment requested at all (MASKARA_PAYEE_ADDRESS unset) — pass-through.
        receipt = _maybe_receipt(resp, HEADER_RECEIPT_V2) or _maybe_receipt(resp, HEADER_RECEIPT_V1)
        _print_result(resp, receipt, paid=receipt is not None)
        return True
    if resp.status_code != 402:
        info(f"unexpected status; body: {body_preview(resp)}")
        return False

    # Step 2: pull the requirements for this protocol version.
    challenge = load_challenge(resp, version)
    if challenge is None:
        info(f"the 402 response carries no x402 v{version} challenge "
             f"(v{version} needs the {'PAYMENT-REQUIRED header' if version == 2 else '402 body'})")
        return False
    requirement = pick_requirement(challenge.get("accepts"), args.network)
    if requirement is None:
        offered = [f"{r.get('scheme')}/{r.get('network')}" for r in challenge.get("accepts", [])
                   if isinstance(r, dict)]
        info(f"no accepts entry for scheme=exact network={args.network} "
             f"({NETWORKS[args.network]['caip2']}); server offered: {offered or '?'}")
        return False

    pay_to = requirement.get("payTo") or ""
    asset = requirement.get("asset") or ""
    amount_field = "amount" if version == 2 else "maxAmountRequired"  # v1 name differs
    amount_raw = requirement.get(amount_field)
    try:
        value = int(str(amount_raw))  # the exact atomic-units string, as sent
    except (TypeError, ValueError):
        info(f"requirement has no usable {amount_field} (got {amount_raw!r})")
        return False

    usd_price = value / (10 ** USDC_DECIMALS)
    info("offer: scheme=exact network={net} price=${usd:.6f} ({value} USDC atomic units)".format(
        net=requirement.get("network"), usd=usd_price, value=value))
    info(f"payTo={pay_to} asset={asset} maxTimeoutSeconds={requirement.get('maxTimeoutSeconds')}")

    if asset and asset.lower() != NETWORKS[args.network]["usdc"].lower():
        info(f"warning: requirement asset {asset} differs from the known USDC "
             f"{NETWORKS[args.network]['usdc']} on this network")

    # Step 3+4: sign the transfer authorization and retry with the payment header.
    now = int(time.time())
    max_timeout = requirement.get("maxTimeoutSeconds") or 60
    try:
        max_timeout = int(max_timeout)
    except (TypeError, ValueError):
        max_timeout = 60
    valid_after = now - CLOCK_SKEW                                   # tolerate skew
    valid_before = now + max(max_timeout, 300)                       # on-chain deadline
    try:
        signature, authorization = sign_transfer_with_authorization(
            account, requirement, value, valid_after, valid_before
        )
    except Exception as exc:
        info(f"signing failed: {exc.__class__.__name__}: {exc}")
        return False

    if version == 2:
        resource = challenge.get("resource") or {}  # echoed back per the v2 flow
        payment = {
            "x402Version": 2,
            "resource": {
                "url": resource.get("url", url),
                "description": resource.get("description", ""),
                "mimeType": resource.get("mimeType", "application/json"),
            },
            "accepted": requirement,  # the exact requirement object we picked
            "payload": {"signature": signature, "authorization": authorization},
        }
        pay_header, receipt_header = HEADER_PAYMENT_V2, HEADER_RECEIPT_V2
    else:
        payment = {
            "x402Version": 1,
            "scheme": requirement.get("scheme", "exact"),
            "network": requirement.get("network", NETWORKS[args.network]["caip2"]),
            "payload": {"signature": signature, "authorization": authorization},
        }
        pay_header, receipt_header = HEADER_PAYMENT_V1, HEADER_RECEIPT_V1

    try:
        retry = session.post(
            url,
            json=payload,
            headers={**base_headers, pay_header: encode_b64_json(payment)},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        info(f"payment retry failed: {exc.__class__.__name__}: {exc}")
        return False
    info(f"retry with {pay_header} -> {retry.status_code} {retry.reason}")

    # Step 5: verify HTTP 200 and a success:true settlement receipt.
    receipt = _maybe_receipt(retry, receipt_header)
    if retry.status_code != 200:
        info(f"payment not accepted; body: {body_preview(retry)}")
        if receipt is not None:
            info(f"settlement receipt: {json.dumps(receipt)}")
        return False
    if receipt is None:
        info(f"200 OK but no {receipt_header} settlement header — cannot verify settlement")
        info(f"body: {body_preview(retry)}")
        return False
    if receipt.get("success") is not True:
        info(f"settlement failed: errorReason={receipt.get('errorReason')!r} "
             f"transaction={receipt.get('transaction', '')!r}")
        return False

    info(f"settlement: transaction={receipt.get('transaction')} network={receipt.get('network')} "
         f"payer={receipt.get('payer')}")
    return _print_result(retry, receipt, paid=True)


def _maybe_receipt(resp, header_name):
    raw = resp.headers.get(header_name)
    if not raw:
        return None
    try:
        return decode_b64_json(raw)
    except ValueError:
        return None


def _print_result(resp, receipt, paid):
    if receipt is not None:
        info(f"settlement: success=true transaction={receipt.get('transaction')} "
             f"network={receipt.get('network')} payer={receipt.get('payer')}")
    else:
        info("(no settlement receipt: request ran free — payments look disabled)")
    try:
        result = resp.json()
    except ValueError:
        info(f"could not parse result body: {body_preview(resp)}")
        return False
    info(f"redacted_text: {result.get('redacted_text')!r}")
    info(f"usage: {result.get('usage')}")
    return paid or result.get("model") == "somukandula/maskara"


# --- CLI ---------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="x402 payment client for MaskaraPay "
                    "(402 parse -> EIP-3009 sign -> retry -> settlement check).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:

  # full dance for both protocol versions against a local server
  MASKARA_PRIVATE_KEY=0x... python scripts/x402/pay_client.py http://127.0.0.1:8080

  # v2 only, custom text, Base Sepolia (default network)
  python scripts/x402/pay_client.py http://127.0.0.1:8080 --version 2 \\
      --text "Email jane.doe@example.com about the invoice."

  # Base mainnet requirements
  python scripts/x402/pay_client.py https://api.example.com --version both --network base
""",
    )
    parser.add_argument("server_url", help="MaskaraPay base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help=f'input text to redact (default "{DEFAULT_TEXT}")')
    parser.add_argument("--version", default="both", choices=["1", "2", "both"],
                        help="x402 protocol version to exercise (default: both)")
    parser.add_argument("--network", default="base-sepolia", choices=sorted(NETWORKS),
                        help="chain whose requirements/price we accept (default base-sepolia)")
    parser.add_argument("--once", action="store_true",
                        help="run a single attempt for the --version value instead of the "
                             "both-versions loop")
    parser.add_argument("--private-key", default=None,
                        help="payer hex key; prefer MASKARA_PRIVATE_KEY (never printed)")
    args = parser.parse_args(argv)

    if args.once and args.version == "both":
        parser.error("--once needs a single --version (1 or 2), not 'both'")

    account = load_account(args.private_key)
    info(f"payer {account.address}")

    versions = [1, 2] if args.version == "both" else [int(args.version)]
    ok = True
    for version in versions:
        ok = run_flow(args, version, account) and ok
    if not ok:
        info("FAILED: at least one protocol version did not complete")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
