#!/usr/bin/env bash
# Runs as root through AWS Systems Manager. GitHub Actions uploads the verified
# binary to a private S3 release prefix, and this script swaps it in place.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 S3_BINARY_URI SHA256" >&2
  exit 64
fi

artifact_uri=$1
expected_sha256=$2
service_binary=/opt/maskarapay/bin/maskara-service
temporary_binary=$(mktemp /tmp/maskarapay-service.XXXXXX)

cleanup() {
  rm -f "$temporary_binary" "${service_binary}.next"
}
trap cleanup EXIT

aws s3 cp "$artifact_uri" "$temporary_binary"
printf '%s  %s\n' "$expected_sha256" "$temporary_binary" | sha256sum --check --status

install -d -o maskara -g maskara /opt/maskarapay/bin
if [[ -f "$service_binary" ]]; then
  cp --reflink=auto "$service_binary" "${service_binary}.previous"
fi
install -o maskara -g maskara -m 0755 "$temporary_binary" "${service_binary}.next"
mv -f "${service_binary}.next" "$service_binary"

healthcheck() {
  systemctl restart maskarapay
  for _ in {1..20}; do
    if curl --fail --silent --show-error http://127.0.0.1:8080/healthz >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if ! healthcheck; then
  if [[ -f "${service_binary}.previous" ]]; then
    echo "new release failed health check; restoring previous binary" >&2
    install -o maskara -g maskara -m 0755 \
      "${service_binary}.previous" "${service_binary}.next"
    mv -f "${service_binary}.next" "$service_binary"
    systemctl restart maskarapay || true
  fi
  journalctl -u maskarapay --no-pager -n 100 >&2
  exit 1
fi

# The Tunnel reconnects automatically, but restart an installed connector after
# a release because this deployment intentionally treats it as the proxy.
if systemctl cat cloudflared >/dev/null 2>&1; then
  systemctl restart cloudflared
  systemctl is-active --quiet cloudflared
fi

echo "MaskaraPay deployment completed"
