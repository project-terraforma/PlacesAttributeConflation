import unittest

from places_attr_conflation.evidence import evidence_from_page, evidence_from_source_type


class EvidenceTests(unittest.TestCase):
    def test_evidence_from_page_uses_ranked_source_score(self):
        item = evidence_from_page(
            "https://city.gov/license",
            "phone",
            "8315551212",
            query='"example" phone',
            page_text="Contact us phone address schema.org",
            recency_days=14,
            zombie_score=0.1,
        )
        self.assertEqual(item.source_type, "government")
        self.assertGreater(item.score(), 0.9)

    def test_evidence_from_source_type_keeps_explicit_type(self):
        item = evidence_from_source_type("government", "https://city.gov/license", "website", "example.com")
        self.assertEqual(item.source_type, "government")
        self.assertAlmostEqual(item.score(), 0.95, places=2)

    def test_evidence_from_source_type_can_capture_freshness(self):
        item = evidence_from_source_type(
            "official_site",
            "https://example.com/contact",
            "website",
            "example.com",
            recency_days=400,
            zombie_score=0.9,
            identity_change_score=0.5,
        )
        self.assertLess(item.score(), 0.8)


if __name__ == "__main__":
    unittest.main()
