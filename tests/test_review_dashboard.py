from places_attr_conflation.review_dashboard import build_review_dashboard_html
from places_attr_conflation.replay import ReplayEpisode


def test_review_dashboard_uses_data_keys_not_fragile_inline_quote_escaping() -> None:
    episode = ReplayEpisode(
        case_id="case-1",
        attribute="website",
        place={"name": "Example"},
        gold_value="https://example.com",
        search_attempts=[],
    )

    html = build_review_dashboard_html([episode])

    assert 'data-key="website_label"' in html
    assert 'data-key="identity_label"' in html
    assert 'data-key="expected_abstain"' in html
    assert "setField(\\'" not in html
