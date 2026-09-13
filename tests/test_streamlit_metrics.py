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


def _llm_sidebar_fields():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "markdown" or not node.args:
            continue
        rendered = node.args[0]
        if not isinstance(rendered, ast.JoinedStr):
            continue
        literal_text = "".join(
            value.value
            for value in rendered.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
        if "LLM:" not in literal_text:
            continue
        return {
            child.slice.value
            for child in ast.walk(rendered)
            if isinstance(child, ast.Subscript)
            and isinstance(child.slice, ast.Constant)
            and isinstance(child.slice.value, str)
        }
    return set()


def test_review_ui_only_renders_metrics_computed_for_that_workflow():
    assert _rendered_metric_labels() == [
        "TS (Trustworthiness)",
        "ATD (Temporal Range)",
    ]


def test_sidebar_identifies_active_llm_provider_and_model():
    assert _llm_sidebar_fields() >= {"provider", "model"}
