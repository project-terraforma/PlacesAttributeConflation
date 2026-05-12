import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConflictDorkTests(unittest.TestCase):
    def test_conflict_dorks_command_exports_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            conflict_csv = Path(tmpdir) / "conflicts.csv"
            conflict_csv.write_text(
                "id,base_id,attribute,truth,truth_source,prediction,baseline,correct,needs_evidence,current_value,base_value\n"
                "case-1,base-1,website,https://good.example,current,https://bad.example,hybrid,false,true,https://bad.example,https://good.example\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "python3",
                    "scripts/run_harness.py",
                    "conflict-dorks",
                    "--conflicts",
                    str(conflict_csv),
                    "--max-queries",
                    "4",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertGreater(payload["rows"], 0)
            self.assertTrue(Path(payload["output_csv"]).exists())


if __name__ == "__main__":
    unittest.main()

