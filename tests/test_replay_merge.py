from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from places_attr_conflation.collector import build_seed_replay_episodes, write_seed_replay_from_batch
from places_attr_conflation.harness import merge_replay_corpora, replay_stats
from places_attr_conflation.replay import load_replay_corpus


HEADER = (
    "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,"
    "current_value,base_value,preferred_sources,layer,query,priority\n"
)


def _payload(text: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "episodes": [
            {
                "case_id": "case-1",
                "attribute": "website",
                "place": {"name": "One"},
                "gold_value": "https://official.example",
                "search_attempts": [
                    {
                        "layer": "official",
                        "query": '"One" official',
                        "fetched_pages": [
                            {
                                "url": "https://official.example",
                                "title": "One",
                                "page_text": text,
                                "source_type": "official_site",
                                "extracted_values": {"website": "https://official.example"},
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _write_batch(path: Path) -> None:
    path.write_text(
        HEADER
        + "case-1,base-1,website,https://official.example,base,https://old.example,hybrid,False,True,"
        + "https://old.example,https://official.example,official_site,official,\"z query\",baseline_wrong\n"
        + "case-1,base-1,website,https://official.example,base,https://old.example,hybrid,False,True,"
        + "https://old.example,https://official.example,official_site,corroboration,\"a query\",baseline_wrong\n"
        + "case-1,base-1,name,Official Name,base,Old Name,hybrid,False,True,"
        + "Old Name,Official Name,official_site,official,\"name query\",baseline_wrong\n",
        encoding="utf-8",
    )


class ReplayMergeTest(unittest.TestCase):
    def test_merge_dedupes_episode_attempt_and_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(__import__("json").dumps(_payload("short")), encoding="utf-8")
            (root / "b.json").write_text(__import__("json").dumps(_payload("longer page text")), encoding="utf-8")
            output = root / "merged.json"

            report = merge_replay_corpora(root, output)
            episodes = load_replay_corpus(output)

            self.assertEqual(report["input_files"], 2)
            self.assertEqual(report["input_episodes"], 2)
            self.assertEqual(report["merged_episodes"], 1)
            self.assertEqual(report["deduped_episodes"], 1)
            self.assertEqual(report["merged_pages"], 1)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0].search_attempts[0].fetched_pages[0].page_text, "longer page text")

    def test_replay_stats_are_stable(self) -> None:
        episodes = load_replay_corpus("tests/fixtures/retrieval_replay_sample.json")
        stats = replay_stats(episodes)

        self.assertEqual(stats["episodes_total"], 4)
        self.assertEqual(stats["attempts_total"], 8)
        self.assertEqual(stats["pages_total"], 9)
        self.assertEqual(stats["authoritative_pages"], 3)
        self.assertAlmostEqual(stats["authoritative_pages_rate"], 3 / 9)
        self.assertEqual(stats["episodes_by_attribute"]["website"], 1)

    def test_seed_replay_from_batch_is_schema_valid_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch.csv"
            _write_batch(batch)
            output = root / "seed.json"

            report = write_seed_replay_from_batch(batch, output)
            episodes = load_replay_corpus(output)
            website = next(episode for episode in episodes if episode.attribute == "website")

            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["pages"], 0)
            self.assertEqual(len(episodes), 2)
            self.assertEqual(website.gold_value, "https://official.example")
            self.assertEqual(website.place["current_value"], "https://old.example")
            self.assertEqual([(attempt.layer, attempt.query) for attempt in website.search_attempts], [("corroboration", "a query"), ("official", "z query")])

    def test_merge_reads_seeded_replay_as_nonzero_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch.csv"
            _write_batch(batch)
            write_seed_replay_from_batch(batch, root / "seed.json")

            report = merge_replay_corpora(root, root / "merged.json")

            self.assertEqual(report["input_files"], 1)
            self.assertEqual(report["input_episodes"], 2)
            self.assertEqual(report["merged_episodes"], 2)
            self.assertGreater(report["merged_attempts"], 0)

    def test_seed_replay_imports_optional_evidence_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch.csv"
            evidence = root / "evidence.csv"
            _write_batch(batch)
            evidence.write_text(
                "case_id,attribute,layer,query,url,title,page_text,source_type,extracted_value\n"
                "case-1,website,official,z query,https://official.example,Official,Official website,official_site,https://official.example\n",
                encoding="utf-8",
            )

            episodes = build_seed_replay_episodes(batch, evidence_path=evidence)
            website = next(episode for episode in episodes if episode.attribute == "website")
            official_attempt = next(attempt for attempt in website.search_attempts if attempt.layer == "official")

            self.assertEqual(len(official_attempt.fetched_pages), 1)
            self.assertEqual(official_attempt.fetched_pages[0].source_type, "official_site")
            self.assertEqual(official_attempt.fetched_pages[0].extracted_values["website"], "https://official.example")


if __name__ == "__main__":
    unittest.main()
