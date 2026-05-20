#!/usr/bin/env python3
"""Populate evidence CSV rows from explicit public URLs.

This script intentionally works from URLs that are already present in evidence
workplans. It is low-volume and replay-oriented: the output must preserve the
original CSV shape while optionally adding website evidence features useful for
PAC replay cases.
"""

from __future__ import annotations

import argparse
import csv
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from places_attr_conflation.website_evidence import clean_html_text, enrich_website_evidence, extract_title


BASE_EXTENDED_FIELDS = [
    "final_url",
    "redirected",
    "http_status",
    "content_hash",
    "canonical_url",
    "domain",
    "registered_domain",
    "page_text_full_path",
    "page_text_excerpt",
    "schema_org_detected",
    "localbusiness_schema_detected",
    "detected_phone",
    "detected_address",
    "detected_name",
    "detected_status",
    "identity_claims",
    "fetch_status",
    "fetch_error",
]


def _fetch(url: str, timeout: float, excerpt_chars: int) -> dict[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MLAttributes public evidence replay/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(max(32768, excerpt_chars * 2))
            text = raw.decode(response.headers.get_content_charset() or "utf-8", errors="replace")
            final_url = str(response.geturl() or url)
            status = str(getattr(response, "status", 200))
            features = enrich_website_evidence(
                requested_url=url,
                final_url=final_url,
                html_text=text,
                http_status=status,
                excerpt_chars=excerpt_chars,
            ).to_dict()
            title = extract_title(text, final_url)
            return {
                "status": "ok",
                "title": title,
                "page_text": clean_html_text(text)[: min(excerpt_chars, 1200)],
                "raw_text": clean_html_text(text),
                "error": "",
                **features,
            }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        parsed = urlparse(url)
        return {
            "status": "error",
            "title": parsed.netloc,
            "page_text": "",
            "raw_text": "",
            "error": str(exc)[:220],
            "final_url": url,
            "redirected": "false",
            "http_status": "",
            "content_hash": "",
            "canonical_url": "",
            "domain": parsed.netloc.lower().removeprefix("www."),
            "registered_domain": parsed.netloc.lower().removeprefix("www."),
            "page_text_excerpt": "",
            "schema_org_detected": "false",
            "localbusiness_schema_detected": "false",
            "detected_phone": "",
            "detected_address": "",
            "detected_name": parsed.netloc,
            "detected_status": "unknown",
            "identity_claims": "",
        }


def _update_notes(notes: str, result: dict[str, str]) -> str:
    import re

    notes = re.sub(r";? ?fetch_status=[^;]+", "", notes or "").strip("; ")
    notes = re.sub(r";? ?http_status=[^;]+", "", notes).strip("; ")
    notes = re.sub(r";? ?fetch_error=[^;]+", "", notes).strip("; ")
    notes = re.sub(r";? ?curl_error=[^;]+", "", notes).strip("; ")
    parts = [part for part in [notes, f"fetch_status={result['status']}"] if part]
    if result.get("http_status"):
        parts.append(f"http_status={result['http_status']}")
    if result.get("error"):
        parts.append(f"fetch_error={result['error']}")
    if result.get("identity_claims"):
        parts.append(f"identity_claims={result['identity_claims']}")
    return "; ".join(parts)


def _fieldnames(existing: list[str], extend_schema: bool) -> list[str]:
    if not extend_schema:
        return existing
    fields = list(existing)
    for field in BASE_EXTENDED_FIELDS:
        if field not in fields:
            fields.append(field)
    return fields


def _write_full_text(full_text_dir: Path | None, csv_path: Path, row_index: int, url: str, text: str) -> str:
    if full_text_dir is None or not text:
        return ""
    full_text_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{csv_path.stem}_{row_index:05d}_{abs(hash(url))}.txt"
    out = full_text_dir / safe_name
    out.write_text(text, encoding="utf-8")
    return str(out)


def populate_csv(
    path: Path,
    timeout: float,
    workers: int,
    *,
    extend_schema: bool = False,
    full_text_dir: Path | None = None,
    excerpt_chars: int = 10000,
) -> dict[str, int | str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    fieldnames = _fieldnames(fieldnames, extend_schema)

    urls = sorted({row.get("url", "") for row in rows if row.get("url", "")})
    results: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, url, timeout, excerpt_chars): url for url in urls}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    ok = 0
    for idx, row in enumerate(rows, start=1):
        url = row.get("url", "")
        if not url:
            continue
        result = results[url]
        if result["status"] == "ok":
            ok += 1
            row["title"] = result["title"] or row.get("title", "")
            row["page_text"] = result["page_text"] or row.get("page_text", "")
        elif not row.get("page_text", ""):
            row["page_text"] = f"Public website URL evidence: {url}"
        row["notes"] = _update_notes(row.get("notes", ""), result)

        if extend_schema:
            for field in BASE_EXTENDED_FIELDS:
                if field == "fetch_status":
                    row[field] = result.get("status", "")
                elif field == "fetch_error":
                    row[field] = result.get("error", "")
                elif field == "page_text_full_path":
                    row[field] = _write_full_text(full_text_dir, path, idx, url, result.get("raw_text", ""))
                else:
                    row[field] = result.get(field, "")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(path), "urls": len(urls), "fetch_ok": ok, "fetch_error": len(urls) - ok}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="Evidence CSV file(s) to update in place.")
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extend-schema", action="store_true", help="Add replay-corpus website enrichment columns.")
    parser.add_argument("--full-text-dir", help="Optional directory for full cleaned page text files.")
    parser.add_argument("--excerpt-chars", type=int, default=10000)
    args = parser.parse_args()
    full_text_dir = Path(args.full_text_dir) if args.full_text_dir else None
    for csv_path in args.csv:
        print(
            populate_csv(
                Path(csv_path),
                timeout=args.timeout,
                workers=args.workers,
                extend_schema=args.extend_schema,
                full_text_dir=full_text_dir,
                excerpt_chars=args.excerpt_chars,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
