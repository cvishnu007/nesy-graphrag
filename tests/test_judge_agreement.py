import pytest

from src.evaluation.judge_agreement import build_prompt, parse_grades, split_items


def test_split_is_deterministic_disjoint_and_balanced():
    ids = [f"R{i:03d}" for i in range(1, 201)]
    first, second = split_items(ids)
    assert split_items(ids) == (first, second)
    assert len(first) == len(second) == 100 and not set(first) & set(second)


def test_parse_grades_validates_count_and_range():
    assert parse_grades('```json\n{"grades": [0, 2]}\n```', 2) == [0, 2]
    with pytest.raises(ValueError):
        parse_grades('{"grades": [0, 3]}', 2)
    with pytest.raises(ValueError):
        parse_grades('{"grades": [1]}', 2)


def test_prompts_hide_labels_and_keep_order():
    prompt = build_prompt("strict", "q", [{"title": "A", "abstract": "x"}, {"title": "B", "abstract": "y"}])
    assert '"position": 1' in prompt and '"position": 2' in prompt and "ai_grade" not in prompt
