from places_attr_conflation.website_evidence import (
    clean_html_text,
    detect_identity_claims,
    detect_schema_org,
    detect_status,
    enrich_website_evidence,
)


def test_enrich_website_evidence_extracts_core_features() -> None:
    html = """
    <html><head>
      <title>New Demo Cafe</title>
      <link rel="canonical" href="https://demo.example/contact" />
      <script type="application/ld+json">{"@type":"LocalBusiness"}</script>
    </head><body>
      <h1>New Demo Cafe</h1>
      <p>Call (831) 555-0199. Visit us at 123 Ocean St.</p>
      <p>We moved to a new location and are now open.</p>
    </body></html>
    """

    features = enrich_website_evidence(
        requested_url="https://old.example",
        final_url="https://demo.example/contact",
        html_text=html,
        http_status=200,
    )

    assert features.final_url == "https://demo.example/contact"
    assert features.redirected == "true"
    assert features.http_status == "200"
    assert features.canonical_url == "https://demo.example/contact"
    assert features.domain == "demo.example"
    assert features.registered_domain == "demo.example"
    assert features.schema_org_detected == "true"
    assert features.localbusiness_schema_detected == "true"
    assert features.detected_phone == "(831) 555-0199"
    assert features.detected_address == "123 Ocean St"
    assert features.detected_status == "moved"
    assert "MOVED" in features.identity_claims
    assert features.content_hash


def test_clean_html_text_removes_scripts_and_tags() -> None:
    assert clean_html_text("<script>bad()</script><p>Hello&nbsp;world</p>") == "Hello world"


def test_detect_identity_claims_and_status() -> None:
    text = "New Cafe, formerly Old Cafe, is under new ownership and temporarily closed."

    claims = detect_identity_claims(text)

    assert "FORMERLY_KNOWN_AS" in claims
    assert "UNDER_NEW_OWNERSHIP" in claims
    assert "TEMPORARILY_CLOSED" in claims
    assert detect_status(text) == "temporarily_closed"


def test_detect_schema_org_requires_schema_signal_for_localbusiness() -> None:
    assert detect_schema_org("LocalBusiness only") == (False, False)
    assert detect_schema_org('<script type="application/ld+json">{"@type":"LocalBusiness"}</script>') == (True, True)
