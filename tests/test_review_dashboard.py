from places_attr_conflation.review_dashboard import build_review_dashboard_html
from places_attr_conflation.replay import ReplayEpisode


def test_review_dashboard_uses_dataset_key_bindings_not_fragile_inline_quote_escaping() -> None:
    episode = ReplayEpisode(
        case_id="case-1",
        attribute="website",
        place={"name": "Example"},
        gold_value="https://example.com",
        search_attempts=[],
    )

    html = build_review_dashboard_html([episode])

    assert "setField(this.dataset.key" in html
    assert "data-key=" in html
    assert "setField(\\'" not in html
