import csv
import json

import pytest

from src.evaluation.human_agreement import main, weighted_kappa


def test_weighted_kappa_extremes():
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    assert weighted_kappa([0, 0, 2, 2], [2, 2, 0, 0]) < 0
    with pytest.raises(ValueError):
        weighted_kappa([], [])


def test_agreement_report_end_to_end(tmp_path):
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"items": [{"item_id": "R1", "ai_grade": 2}, {"item_id": "R2", "ai_grade": 0},
                                         {"item_id": "R3", "ai_grade": 1}]}))
    paths = []
    for name, grades in (("A", ["2", "0", "1"]), ("B", ["2", "0", "0"])):
        path = tmp_path / f"sheet_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["item_id", "grade_0_1_2"])
            writer.writeheader()
            writer.writerows({"item_id": f"R{i}", "grade_0_1_2": g} for i, g in enumerate(grades, 1))
        paths.append(str(path))
    output = tmp_path / "report.json"
    main(["--sheets", *paths, "--key", str(key), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["human_vs_ai"]["sheet_A"]["exact_agreement"] == 1.0
    assert report["ai_vs_consensus"]["consensus_items"] == 2
