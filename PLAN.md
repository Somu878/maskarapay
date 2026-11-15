# MaskaraPay — execution plan (verified 15 Nov 2026)

Rule for this file and every claim made anywhere in the repo: **only claim what is
built AND tested.** Nothing here funds wallets, buys services, or deploys paid
infrastructure without Somu's explicit approval.

## 1. Where the project actually stands (verified in the repo, 15 Nov 2026)

Committed baseline (`78b4d02`, 5 Nov):
- `POST /api/v1/redact` + `GET /healthz` (Axum 0.8), pipeline with 3 unit tests —
  `cargo test --release` green (re-ran today: 3 passed).
- EC2 + Cloudflare Tunnel deploy files and runbooks.
- Public endpoint `api.somuso.fun` was down on 8 Nov — re-check before relying on it.

Committed this session (`9b72de9`) — client side built, **not yet exercised against a
paying server**:
- `scripts/x402/pay_client.py` — full 402 dance client (v1 + v2 headers, EIP-3009
  EIP-712 signing via `eth-account`, receipt decoding).
- `scripts/x402/mock_facilitator.py` — stdlib-only facilitator stand-in with
  failure switches (`MOCK_VERIFY_FAIL=1`, `MOCK_SETTLE_FAIL=1`).
- `docs/x402.md` — full contract: env vars, flows, payload reference.
- README + systemd units document the payment env vars.
- `.gitignore` now also excludes `.env` (keys must never be committed).

**M1 completed:** the Rust server-side 402 gate is implemented in
`service/src/payment.rs` and is wired into `service/src/main.rs`. The complete
M1 test matrix passes locally; payment remains disabled until
`MASKARA_PAYEE_ADDRESS` is set.

## 2. Protocol facts verified today against official docs (docs.x402.org + x402-foundation specs)

- v2 headers: `PAYMENT-REQUIRED` / `PAYMENT-SIGNATURE` / `PAYMENT-RESPONSE`,
  all base64-encoded JSON — matches the existing client exactly.
- v1 legacy: `X-PAYMENT` / `X-PAYMENT-RESPONSE`, 402 body carries the challenge,
  `maxAmountRequired` naming, plain network names.
- Error mapping: malformed payment → 400; verify/settle failure → 402 (failure
  receipt header); facilitator outage → 500; success → 200.
- Networks: Base Sepolia `eip155:84532`, Base mainnet `eip155:8453`.
- USDC Base Sepolia `0x036CbD53842c5426634e7929541eC2318f3dCF7e`, EIP-712 domain
  `USDC`/`2` (from the official exact-EVM spec examples). Base mainnet USDC
  `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`, domain `USD Coin`/`2` — re-verify
  the mainnet pair on the day of M4.
- `x402.org/facilitator` is testnet-only per official docs. Mainnet options:
  CDP (`https://api.cdp.coinbase.com/platform/v2/x402`, needs keys), PayAI (no
  API keys), Mogami (Base), Dexter, and others on the facilitators page.
- `x402-axum` crate: not adopted. The hand-written Axum middleware fallback was
  chosen so the server matches the already-built client/docs contract.

## 3. Milestones (in order; M1–M3 = testnet checkpoint, M4 = final)

### M1 — Local 402 gate (no money, no accounts) — DONE (commit `2520e1a`, 15 Nov 2026)
Server-side gate in `service/src/payment.rs`, wired into `main.rs`:
- `MASKARA_PAYEE_ADDRESS` unset → everything free (current behavior preserved).
- Set on `/api/v1/redact` only; `/healthz` stays free.
- Settle-first ordering; fail-closed on facilitator outage; in-memory replay
  cache (nonce) so a retried payment can neither double-charge nor yield a free
  result; v2 + v1 dialects; env-var config only, no keys in the repo.
Acceptance: the full M1 test matrix in Rust passes (402 without payment,
malformed payment, wrong amount/network/asset/payTo, replay, settle-fail,
verify-fail, facilitator-down, healthz-free, v1+v2 happy paths, price parsing),
plus the existing pipeline tests.

### M2 — Real payment on Base Sepolia (testnet, free) — READY, awaiting go-ahead
- Wallets exist locally (keys in `~/.maskarapay-testnet/`, never printed/committed):
  seller `0x179A0fcCF2Fdc7875f0bB66dBc8001f6b9e490F3` (receives), buyer
  `0xb0c8fDa371a39a6D61597C1E6301b9F8E227214e` — funded on-chain with 20.0 USDC
  (verified via RPC 15 Nov). Circle faucet (faucet.circle.com → USDC → Base
  Sepolia) is the verified funding source.
- Live facilitator verified: `https://x402.org/facilitator/supported` serves
  v2 `exact` on `eip155:84532` and v1 `exact` on `base-sepolia`.
- ~~pay_client.py broken against all current `eth-account` releases~~ — fixed
  (commit `db66c46`, compat shim, pinned `eth-account==0.14.0`, E2E proven
  locally on v1+v2 with throwaway keys).
Acceptance: pay client prints redacted text + receipt with a tx hash visible on
a Base Sepolia explorer. Somu's go-ahead required for the real paid call.

### M3 — Deploy and prove — needs Somu's EC2/Cloudflare access
- Bring the existing EC2 + Tunnel deployment back up; confirm
  `https://api.somuso.fun/healthz`.
- Buyer script against the public URL on Base Sepolia; save output + tx link.
- README updated with only-true claims: how to run/pay, price, network, honest
  limits. One-page 402 flow diagram in `docs/`.
Acceptance (checkpoint): tests green incl. 402-without-payment; public paid call
prints receipt with explorer-visible tx; truthful README + demo output.

### M4 — Base mainnet (FINAL TARGET) — gated on ALL of
1. Chosen facilitator currently supports Base mainnet (checked that day).
2. Current fees/free-tier/minimums written down before going live.
3. Somu's explicit approval: which wallets, how much he funds, the price
   (`MASKARA_PRICE_USD` is already a config value; ~$0.001 was a proposal, his call).
Then: `MASKARA_X402_NETWORK=base` + mainnet facilitator, one real paid call with
the buyer script, tx recorded, README updated to mainnet.

## 4. Honest resume wording
- After M1–M3 checkpoint: "x402 pay-per-call PII redaction API (Base Sepolia testnet)".
- "mainnet" only after M4 has a real mainnet tx.
- Grafana/Prometheus dashboards: not built, not claimed, optional.

## 5. Standing rules
- Never log request text or redacted output (repo rule; payment code follows it too).
- Private keys only in a gitignored `.env` / shell env, never in repo or chat.
- No funding, purchases, or paid-infra deploys without Somu's confirmation.
- Do not push to GitHub without Somu's go-ahead (the CI/CD runbook deploys on push).
