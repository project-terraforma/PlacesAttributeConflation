"""Website evidence enrichment helpers for replay-corpus construction."""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from urllib.parse import urlparse


IDENTITY_CLAIM_PATTERNS = {
    "FORMERLY_KNOWN_AS": r"\bformerly\b|\bformer(?:ly)? known as\b",
    "MOVED": r"\bwe moved\b|\bmoved to\b|\bnew location\b",
    "UNDER_NEW_OWNERSHIP": r"\bunder new ownership\b",
    "REPLACED_BY": r"\breplaced by\b|\bis now\b",
    "PERMANENTLY_CLOSED": r"\bpermanently closed\b|\bclosed permanently\b",
    "TEMPORARILY_CLOSED": r"\btemporarily closed\b|\btemporary closure\b",
    "GRAND_OPENING": r"\bgrand opening\b|\bnow open\b",
}

PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
ADDRESS_RE = re.compile(r"\b\d{1,6}\s+[A-Za-z0-9 .'-]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court|Pl|Place)\b", re.IGNORECASE)
CANONICAL_RE = re.compile(r"(?is)<link[^>]+rel=[\"']?canonical[\"']?[^>]+href=[\"']([^\"']+)[\"']")
TITLE_RE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
JSONLD_RE = re.compile(r"(?is)<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>")
META_SITE_NAME_RE = re.compile(r"(?is)<meta[^>]+property=[\"']og:site_name[\"'][^>]+content=[\"']([^\"']+)[\"']")


@dataclass(frozen=True)
class WebsiteEvidenceFeatures:
    final_url: str = ""
    redirected: str = ""
    http_status: str = ""
    content_hash: str = ""
    canonical_url: str = ""
    domain: str = ""
    registered_domain: str = ""
    page_text_excerpt: str = ""
    schema_org_detected: str = ""
    localbusiness_schema_detected: str = ""
    detected_phone: str = ""
    detected_address: str = ""
    detected_name: str = ""
    detected_status: str = "unknown"
    identity_claims: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def clean_html_text(value: str) -> str:
    value = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", value or "")
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def domain_from_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.netloc.lower().removeprefix("www.")


def registered_domain(domain: str) -> str:
    parts = [part for part in domain.lower().split(".") if part]
    if len(parts) <= 2:
        return domain.lower()
    return ".".join(parts[-2:])


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def extract_canonical_url(html_text: str) -> str:
    match = CANONICAL_RE.search(html_text or "")
    return html.unescape(match.group(1).strip()) if match else ""


def extract_title(html_text: str, fallback_url: str = "") -> str:
    match = TITLE_RE.search(html_text or "")
    if match:
        return clean_html_text(match.group(1))[:180]
    return domain_from_url(fallback_url)


def detect_schema_org(html_text: str) -> tuple[bool, bool]:
    lowered = (html_text or "").lower()
    schema = "schema.org" in lowered or bool(JSONLD_RE.search(html_text or ""))
    local = any(token in lowered for token in ("localbusiness", "restaurant", "store", "organization")) and schema
    return schema, local


def detect_identity_claims(text: str) -> list[str]:
    lowered = (text or "").lower()
    return [label for label, pattern in IDENTITY_CLAIM_PATTERNS.items() if re.search(pattern, lowered)]


def detect_status(text: str) -> str:
    lowered = (text or "").lower()
    if "permanently closed" in lowered or "closed permanently" in lowered:
        return "permanently_closed"
    if "temporarily closed" in lowered or "temporary closure" in lowered:
        return "temporarily_closed"
    if "we moved" in lowered or "moved to" in lowered or "new location" in lowered:
        return "moved"
    if "now open" in lowered or "open now" in lowered or "hours" in lowered:
        return "active"
    return "unknown"


def extract_detected_name(html_text: str, title: str) -> str:
    match = META_SITE_NAME_RE.search(html_text or "")
    if match:
        return clean_html_text(match.group(1))[:180]
    return title[:180]


def extract_first(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text or "")
    return match.group(0).strip() if match else ""


def enrich_website_evidence(
    *,
    requested_url: str,
    final_url: str,
    html_text: str,
    http_status: int | str = "",
    excerpt_chars: int = 10000,
) -> WebsiteEvidenceFeatures:
    clean_text = clean_html_text(html_text)
    title = extract_title(html_text, final_url or requested_url)
    schema, local = detect_schema_org(html_text)
    domain = domain_from_url(final_url or requested_url)
    claims = detect_identity_claims(clean_text)
    return WebsiteEvidenceFeatures(
        final_url=final_url or requested_url,
        redirected=str(bool(final_url and final_url != requested_url)).lower(),
        http_status=str(http_status or ""),
        content_hash=content_hash(clean_text),
        canonical_url=extract_canonical_url(html_text),
        domain=domain,
        registered_domain=registered_domain(domain),
        page_text_excerpt=clean_text[: max(0, excerpt_chars)],
        schema_org_detected=str(schema).lower(),
        localbusiness_schema_detected=str(local).lower(),
        detected_phone=extract_first(PHONE_RE, clean_text),
        detected_address=extract_first(ADDRESS_RE, clean_text),
        detected_name=extract_detected_name(html_text, title),
        detected_status=detect_status(clean_text),
        identity_claims=",".join(claims),
    )
