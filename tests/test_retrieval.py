import unittest

from src.pipeline.retrieval import build_retrieval_diagnostics, fuse_results


def paper(paper_id, score=0.8):
    return {
        "id": paper_id,
        "title": paper_id,
        "abstract": "",
        "year": 2024,
        "category": "Computer Science",
        "score": score,
    }


class RetrievalFusionTests(unittest.TestCase):
    def test_graph_only_results_can_enter_top_k(self):
        neural = [paper(f"n{i}", 1.0 - i / 100) for i in range(1, 11)]
        symbolic = [paper(f"s{i}", 1.0 - i / 100) for i in range(1, 11)]

        results = fuse_results(neural, symbolic, top_k=10)

        self.assertEqual(len(results), 10)
        self.assertTrue(any(item["source"] == "symbolic" for item in results))
        self.assertTrue(any(item["source"] == "neural" for item in results))

    def test_overlap_is_labelled_both_and_boosted(self):
        neural = [paper("shared"), paper("neural-only")]
        symbolic = [paper("shared"), paper("graph-only")]

        results = fuse_results(neural, symbolic, top_k=3)

        self.assertEqual(results[0]["id"], "shared")
        self.assertEqual(results[0]["source"], "both")
        self.assertIn("neural_rank", results[0])
        self.assertIn("graph_rank", results[0])
        self.assertLessEqual(results[0]["score"], 1.0)

    def test_fusion_does_not_mutate_input_results(self):
        neural = [paper("n1")]
        symbolic = [paper("s1")]

        fuse_results(neural, symbolic, top_k=2)

        self.assertNotIn("neural_rank", neural[0])
        self.assertNotIn("graph_rank", symbolic[0])

    def test_diagnostics_explain_kept_and_dropped_candidates(self):
        neural = [paper("shared"), paper("neural-only")]
        symbolic = [
            {**paper("shared"), "graph_connections": 3},
            {**paper("graph-only"), "graph_connections": 2},
            {**paper("dropped"), "graph_connections": 1},
        ]
        final = fuse_results(neural, symbolic, top_k=2)

        report = build_retrieval_diagnostics(
            neural,
            symbolic,
            final,
            {"shared": 4, "neural-only": 0, "graph-only": 2, "dropped": 1},
        )
        rows = {row["id"]: row for row in report["candidates"]}

        self.assertEqual(report["source_distribution"]["both"], 1)
        self.assertEqual(rows["shared"]["citation_degree"], 4)
        self.assertEqual(rows["shared"]["decision"], "kept in final top-k")
        self.assertEqual(rows["dropped"]["decision"], "dropped below final cutoff")


if __name__ == "__main__":
    unittest.main()
