# AWS EC2 + Cloudflare Tunnel deployment

This runbook deploys the API without Route 53, an Elastic IP, an Application
Load Balancer, or inbound EC2 firewall rules. Cloudflare remains authoritative
for the domain and provides the public hostname, TLS, and initial access
control. The EC2 instance only makes outbound connections.

```text
https://api.example.com/api/v1/redact
        |
Cloudflare DNS + Access + Tunnel
        |
AWS EC2: cloudflared -> 127.0.0.1:8080 -> MaskaraPay
```

Replace `api.example.com` with the API subdomain in the Cloudflare zone. Do not
put Cloudflare tunnel tokens, AWS credentials, or model binaries in this
repository.

## 1. Cost and account guardrails

Create AWS Budgets alerts before launching anything, for example at USD 5 and
USD 15. Alerts do not stop usage, but reveal unexpected resources promptly.

Use a single x86_64 Ubuntu 24.04 EC2 instance. `t3.small` (2 GiB RAM) is the
recommended starting size for the model. If the account only permits a 1 GiB
micro instance, do not build the Rust release binary there; build it elsewhere
and copy the binary to the server, then verify the model can run within memory.

For the simplest first deployment, the instance needs outbound Internet access
for Systems Manager, package installation, GitHub, Hugging Face, and
Cloudflare Tunnel. Launching it in a public subnet with no inbound rules meets
that need, but an AWS public IPv4 address is charged separately at USD
0.005/hour (about USD 3.60 for a 30-day month). This charge is independent of
EC2 Free Tier eligibility.
Do not create a NAT Gateway for this small service: it costs far more. An
IPv6-only/private design can avoid the IPv4 charge, but needs additional VPC
and Systems Manager endpoint planning and is deliberately not the first setup.

## 2. Model artifacts are fetched on the instance

The repository deliberately ignores `assets/*.onnx`. Do not pre-stage those
files in S3. Both the development and production services check for a local
runtime model on startup. If `maskara-int8.onnx` and `maskara-f32.onnx` are
missing, the process runs `deploy/ensure-model.sh` on that machine, which
calls `export/download_model.py` and then exports ONNX.

The first start therefore needs RAM for the Python export (prefer `t3.small`
or larger) and outbound access to Hugging Face. Later restarts skip the
download when the ONNX files are already on disk. The bootstrap explicitly
uses Python 3.12: the pinned PyTorch version does not support Ubuntu 26.04's
default Python 3.14.

## 3. Launch the EC2 instance

In the AWS console, launch the selected Ubuntu 24.04 x86_64 instance in one
Region and configure:

- Attach an instance role with the `AmazonSSMManagedInstanceCore` managed
  policy.
- Require IMDSv2.
- Create a security group with **no inbound rules**. Keep the default outbound
  rule so the instance can connect to Cloudflare, SSM, Hugging Face, and
  package sources.
- Do not open ports 22, 80, 443, or 8080. Use Systems Manager Session Manager
  instead of SSH.

Wait until the instance appears as a managed node in Systems Manager, then
open a Session Manager shell. Systems Manager can use either the recommended
account-level host management configuration or an instance profile.

## 4. Install the release

Run the following from the Session Manager shell. The Rust toolchain is needed
only while building this release; the service runs the compiled binary.

```bash
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl git libssl-dev pkg-config python3 python3-venv python3-pip
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
sudo install -m 0755 "$HOME/.local/bin/uv" /usr/local/bin/uv

git clone https://github.com/Somu878/maskarapay.git /tmp/maskarapay-source
cd /tmp/maskarapay-source/service
cargo build --release --locked

sudo adduser --system --group --home /opt/maskarapay maskara
sudo install -d -o maskara -g maskara /opt/maskarapay/bin /opt/maskarapay/assets \
  /opt/maskarapay/export
sudo install -o maskara -g maskara -m 0755 target/release/service \
  /opt/maskarapay/bin/maskara-service
sudo install -o maskara -g maskara -m 0755 ../deploy/ensure-model.sh \
  /opt/maskarapay/bin/ensure-model
sudo install -o maskara -g maskara -m 0644 ../assets/config.json \
  ../assets/tokenizer.json /opt/maskarapay/assets/
sudo cp -a ../export/. /opt/maskarapay/export/
sudo chown -R maskara:maskara /opt/maskarapay/export
```

Copy the tracked systemd unit for this instance. Use
`deploy/maskarapay.service` on production and `deploy/maskarapay-dev.service`
on the development instance. First start may take several minutes while the
instance downloads Hugging Face weights and exports ONNX.

```bash
sudo install -m 0644 ../deploy/maskarapay.service /etc/systemd/system/maskarapay.service
# development instance instead:
# sudo install -m 0644 ../deploy/maskarapay-dev.service /etc/systemd/system/maskarapay.service
sudo systemctl daemon-reload
sudo systemctl enable --now maskarapay
curl --fail-with-body http://127.0.0.1:8080/healthz
```

If the service does not start, inspect only metadata-safe logs:

```bash
sudo journalctl -u maskarapay --no-pager -n 100
```

## 5. Create the Cloudflare Tunnel and hostname

In the Cloudflare dashboard:

1. Go to **Networking > Tunnels**, create a remotely-managed tunnel named
   `maskarapay-production`, choose Linux, and run the generated install command
   on the EC2 instance. Treat the command's tunnel token like a password: do
   not paste it into chat, commit it, or save it in shell history.
2. Wait for the tunnel to show **Healthy**.
3. Add a **Published application** route:
   - Hostname: `api.example.com`
   - Service URL: `http://127.0.0.1:8080`

Cloudflare creates the DNS route in the existing Cloudflare zone. Route 53 is
not involved. Cloudflare terminates browser/client TLS; the origin remains
loopback-only on EC2.

## 6. Restrict the first public API

Before treating the hostname as available, create a Cloudflare Zero Trust
Access self-hosted application for `api.example.com` and add a service-token
policy. This keeps the API private to the machine or integration holding that
token. The current Rust service has no application-level API-key validation,
so do not create an Allow Everyone policy.

## 7. Verify the deployed API

Create a Cloudflare Access service token and run the following from the
authorized caller. Keep both values in its secret store, not a source file.

```bash
curl --fail-with-body -sS https://api.example.com/api/v1/redact \
  -H 'content-type: application/json' \
  -H 'CF-Access-Client-Id: YOUR_SERVICE_TOKEN_ID' \
  -H 'CF-Access-Client-Secret: YOUR_SERVICE_TOKEN_SECRET' \
  --data '{
    "model": "somukandula/maskara",
    "text": "Email ravi.sharma@example.com about the invoice."
  }'
```

Expect `ravi.sharma@example.com` to be replaced with `[EMAIL]`. Also check
`https://api.example.com/healthz` with the same Access headers.

## Updating or rolling back

Build a new release, replace `/opt/maskarapay/bin/maskara-service`, and copy
an updated `export/` tree when the pinned model revision changes. If you need
a fresh download, remove the ONNX files under `/opt/maskarapay/assets` and
run:

```bash
sudo systemctl restart maskarapay
sudo systemctl status maskarapay --no-pager
```

Keep the previous release binary until the health and redaction checks succeed.
Rollback is then copying it back and restarting the service.

## Sources

- [Cloudflare Tunnel setup](https://developers.cloudflare.com/tunnel/get-started/)
- [Cloudflare dashboard tunnel setup](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/)
- [AWS Systems Manager instance permissions](https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-instance-permissions.html)
- [AWS public IPv4 pricing](https://aws.amazon.com/vpc/pricing/)
