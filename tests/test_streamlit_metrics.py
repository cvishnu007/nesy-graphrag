import ast
from pathlib import Path


APP_PATH = Path(__file__).parents[1] / "app" / "streamlit_app.py"


def _rendered_metric_labels():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    labels = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "metric" or not node.args:
            continue
        label = node.args[0]
        if isinstance(label, ast.Constant) and isinstance(label.value, str):
            labels.append(label.value)
    return labels


def test_review_ui_only_renders_metrics_computed_for_that_workflow():
    assert _rendered_metric_labels() == [
        "TS (Trustworthiness)",
        "ATD (Temporal Range)",
    ]
