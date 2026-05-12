import unittest

from places_attr_conflation.manifest import EvidenceItem
from places_attr_conflation.resolver import resolve_attribute


class ResolverDecisionTests(unittest.TestCase):
    def test_resolver_selects_candidate_with_stronger_evidence(self):
        evidence = [
            EvidenceItem("official_site", "https://a.com", "phone", "(831) 555-1212"),
            EvidenceItem("government", "https://city.gov/license", "phone", "8315551212"),
            EvidenceItem("aggregator", "https://yelp.com/biz/a", "phone", "8315559999"),
        ]
        decision = resolve_attribute("phone", ["8315551212", "8315559999"], evidence)
        self.assertFalse(decision.abstained)
        self.assertEqual(decision.decision, "8315551212")

    def test_resolver_abstains_without_matching_evidence(self):
        evidence = [EvidenceItem("official_site", "https://a.com", "website", "other.com")]
        decision = resolve_attribute("website", ["example.com"], evidence)
        self.assertTrue(decision.abstained)

    def test_resolver_abstains_on_low_authority_only_match(self):
        evidence = [EvidenceItem("aggregator", "https://directory.example/biz", "category", "restaurant")]
        decision = resolve_attribute("category", ["restaurant", "bakery"], evidence)
        self.assertTrue(decision.abstained)
        self.assertIn("minimum authority", decision.reason)

    def test_resolver_prefers_fresher_evidence_when_candidates_compete(self):
        evidence = [
            EvidenceItem("official_site", "https://old.example.com/contact", "website", "old.example.com", recency_days=500, zombie_score=0.9),
            EvidenceItem("official_site", "https://fresh.example.com/contact", "website", "fresh.example.com", recency_days=7),
        ]
        decision = resolve_attribute("website", ["old.example.com", "fresh.example.com"], evidence)
        self.assertFalse(decision.abstained)
        self.assertEqual(decision.decision, "fresh.example.com")

    def test_resolver_abstains_when_only_weak_aggregators_support_category(self):
        evidence = [
            EvidenceItem("aggregator", "https://directory1.example/biz", "category", "restaurant"),
            EvidenceItem("aggregator", "https://directory2.example/biz", "category", "restaurant"),
        ]
        decision = resolve_attribute("category", ["restaurant", "bakery"], evidence)
        self.assertTrue(decision.abstained)


if __name__ == "__main__":
    unittest.main()
