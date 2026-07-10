"""Make HTTPS fetches trust the OS certificate store (Windows).

On Windows behind a TLS-inspecting proxy or antivirus, the HTTPS handshake is
re-signed by a local root CA that Python's bundled store (certifi) does not
know, so yfinance/curl_cffi fail with
``curl: (60) SSL certificate problem: unable to get local issuer certificate``
and every price/fundamentals fetch returns empty — silently serving stale
data. This builds a combined bundle (certifi + the Windows ROOT/CA stores)
once per process and points the HTTP stacks at it via the standard env vars.

No-op on non-Windows (e.g. the Linux CI runners) and when the caller has
already set a CA bundle, so it never overrides an explicit configuration.
"""
from __future__ import annotations

import base64
import logging
import os
import ssl
import sys
import tempfile
from pathlib import Path

import certifi

log = logging.getLogger("egx-mcp.certs")

_ENV_VARS = ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


def ensure_ca_bundle() -> str | None:
    """Idempotently ensure a CA bundle covering the OS trust store is active.

    Returns the bundle path, or None if not needed / unavailable."""
    if sys.platform != "win32":
        return None
    if os.environ.get("EGX_CA_BUNDLE_READY"):
        return os.environ.get("CURL_CA_BUNDLE")
    # Respect an existing explicit configuration — don't clobber it.
    if any(os.environ.get(v) for v in _ENV_VARS):
        os.environ["EGX_CA_BUNDLE_READY"] = "1"
        return os.environ.get("CURL_CA_BUNDLE")
    try:
        pem = [Path(certifi.where()).read_text(encoding="utf-8")]
        for store in ("ROOT", "CA"):
            for cert, _enc, _trust in ssl.enum_certificates(store):  # type: ignore[attr-defined]
                b64 = base64.encodebytes(cert).decode("ascii")
                pem.append(f"-----BEGIN CERTIFICATE-----\n{b64}-----END CERTIFICATE-----\n")
        out = Path(tempfile.gettempdir()) / "egx_ca_bundle.pem"
        out.write_text("\n".join(pem), encoding="utf-8")
        for var in _ENV_VARS:
            os.environ[var] = str(out)
        os.environ["EGX_CA_BUNDLE_READY"] = "1"
        log.info("Using OS-trust CA bundle for HTTPS fetches: %s", out)
        return str(out)
    except Exception as e:  # noqa: BLE001
        log.warning("could not build OS CA bundle (%s) — HTTPS fetches may fail behind a proxy", e)
        return None
