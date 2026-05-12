import unittest

from places_attr_conflation.dorking import (
    audit_dorking_plans,
    audit_multi_layer_plan,
    build_multi_layer_plan,
    build_query_plan,
    classify_source,
    loose_query,
    rank_source,
    targeted_queries,
)


class DorkingTests(unittest.TestCase):
    def test_targeted_queries_are_attribute_specific(self):
        place = {"name": "Cafe Rio", "city": "Santa Cruz", "address": "100 Main St", "phone": "8315551212"}
        website_queries = targeted_queries(place, "website")
        self.assertTrue(any("official website" in query for query in website_queries))
        self.assertNotEqual(loose_query(place), website_queries[0])

    def test_source_classifier_separates_gov_social_and_official(self):
        self.assertEqual(classify_source("https://business.ca.gov/example"), "government")
        self.assertEqual(classify_source("https://facebook.com/example"), "social")
        self.assertEqual(classify_source("https://www.google.com/maps/place/example"), "google_places")
        self.assertEqual(classify_source("https://www.openstreetmap.org/node/1"), "osm")
        self.assertEqual(classify_source("https://www.bbb.org/us/ca/example"), "business_registry")
        self.assertEqual(classify_source("https://example.com"), "official_site")

    def test_query_plan_keeps_loose_and_targeted_queries_separate(self):
        plan = build_query_plan({"name": "Cafe Rio", "city": "Santa Cruz", "region": "CA"}, "website")
        self.assertEqual(plan.attribute, "website")
        self.assertEqual(plan.loose, "Cafe Rio Santa Cruz CA")
        self.assertGreaterEqual(len(plan.targeted), 2)
        self.assertIn("official_site", plan.preferred_sources)

    def test_targeted_queries_include_same_domain_verification_for_known_websites(self):
        place = {
            "name": "Cafe Rio",
            "city": "Santa Cruz",
            "region": "CA",
            "address": "100 Main St",
            "phone": "8315551212",
            "website": "https://caferio.example",
        }
        queries = targeted_queries(place, "website")
        self.assertTrue(any("site:caferio.example" in query for query in queries))
        self.assertTrue(any("schema.org" in query for query in queries))

    def test_rank_source_prefers_official_and_page_metadata(self):
        official = rank_source("https://example.com/contact", "Contact us phone address schema.org")
        agg = rank_source("https://yelp.com/biz/example", "Contact us")
        self.assertGreater(official, agg)

    def test_rank_source_penalizes_stale_and_moved_pages(self):
        fresh = rank_source("https://example.com/contact", "Official contact page schema.org updated hours", query='"Cafe Rio" contact')
        stale = rank_source(
            "https://yelp.com/biz/example",
            "Reviews, permanently closed, moved, former location, directory listing",
            query='"Cafe Rio" contact',
        )
        self.assertGreater(fresh, stale)

    def test_multi_layer_plan_has_escalation_layers(self):
        plan = build_multi_layer_plan({"name": "Cafe Rio", "city": "Santa Cruz", "region": "CA", "phone": "8315551212"}, "phone")
        self.assertEqual([layer.name for layer in plan.layers], ["official", "corroboration", "freshness", "fallback"])
        self.assertGreaterEqual(len(plan.layers[0].queries), 2)
        self.assertTrue(any("open now" in query for query in plan.layers[2].queries))
        self.assertTrue(plan.layers[-1].queries[0].startswith("Cafe Rio"))

    def test_targeted_queries_use_search_operators_and_aggregator_exclusions(self):
        place = {
            "name": "Cafe Rio",
            "city": "Santa Cruz",
            "region": "CA",
            "address": "100 Main St",
            "phone": "8315551212",
            "website": "https://caferio.example",
        }
        queries = targeted_queries(place, "website")
        self.assertTrue(any("site:.gov" in query for query in queries))
        self.assertTrue(any("intitle:" in query or "inurl:" in query for query in queries))
        self.assertTrue(any("-site:yelp.com" in query for query in queries))
        self.assertTrue(any('"Cafe Rio"' in query for query in queries))

    def test_dork_plan_audit_measures_operator_quality(self):
        plan = build_multi_layer_plan(
            {
                "name": "Cafe Rio",
                "city": "Santa Cruz",
                "region": "CA",
                "address": "100 Main St",
                "phone": "8315551212",
                "website": "https://caferio.example",
            },
            "address",
        )
        audit = audit_multi_layer_plan(plan)
        self.assertGreaterEqual(audit.operator_coverage, 0.75)
        self.assertGreater(audit.site_restricted_queries, 0)
        self.assertGreater(audit.exclusion_queries, 0)
        self.assertLess(audit.fallback_share, 0.2)

    def test_dorking_audit_report_aggregates_multiple_attributes(self):
        report = audit_dorking_plans(
            [{"name": "Cafe Rio", "city": "Santa Cruz", "region": "CA", "address": "100 Main St", "phone": "8315551212"}],
            ["website", "phone"],
        )
        self.assertEqual(report["totals"]["plans"], 2)
        self.assertGreater(report["summary"]["operator_coverage"], 0.7)
        self.assertGreater(report["summary"]["authority_coverage"], 0.6)


if __name__ == "__main__":
    unittest.main()
