import unittest

from places_attr_conflation.manifest import AttributeDecision, EvidenceItem, EvidenceManifest


class ManifestSchemaTests(unittest.TestCase):
    def test_manifest_serializes_decision_and_evidence(self):
        item = EvidenceItem(
            "official_site",
            "https://example.com",
            "phone",
            "8315551212",
            recency_days=30,
            zombie_score=0.1,
            identity_change_score=0.0,
        )
        decision = AttributeDecision("phone", "8315551212", 1.0, "official site match", [item])
        manifest = EvidenceManifest("poi-1", {"name": "Example"}, [decision])
        payload = manifest.to_dict()
        self.assertEqual(payload["poi_id"], "poi-1")
        self.assertEqual(payload["decisions"][0]["evidence"][0]["source_rank"], 1.0)
        self.assertEqual(payload["decisions"][0]["evidence"][0]["recency_days"], 30)


if __name__ == "__main__":
    unittest.main()
