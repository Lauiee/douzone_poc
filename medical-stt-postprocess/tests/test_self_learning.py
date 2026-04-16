"""자기학습 모듈 스모크 테스트"""

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.self_learning.capture import log_pair
from src.self_learning.aggregate import aggregate


class TestSelfLearning(unittest.TestCase):
    def test_aggregate_min_count(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.jsonl"
            log_pair("극성충수염", "급성충수염", source="human", path=p)
            log_pair("극성충수염", "급성충수염", source="human", path=p)
            log_pair("a", "b", source="human", path=p)  # 한 번만

            pairs = aggregate(p, min_count=2)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].original, "극성충수염")
            self.assertEqual(pairs[0].corrected, "급성충수염")
            self.assertEqual(pairs[0].count, 2)


if __name__ == "__main__":
    unittest.main()
