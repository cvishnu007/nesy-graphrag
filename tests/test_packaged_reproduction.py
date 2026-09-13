from pathlib import Path
import hashlib
import pytest
from src.evaluation.packaged_reproduction import verify_frozen_benchmarks, verify_digest


def test_digest_accepts_only_exact_or_crlf_equivalent_bytes():
    expected = hashlib.sha256(b'a\r\nb\r\n').hexdigest()
    assert verify_digest(b'a\nb\n', expected) == 'line_endings_only'
    with pytest.raises(ValueError, match='hash'):
        verify_digest(b'a\nc\n', expected)


def test_packaged_frozen_benchmarks_are_found_and_validated_outside_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = verify_frozen_benchmarks()
    assert {key: r['count'] for key, r in result.items()} == {'contradiction': 62, 'support': 151, 'hypothesis': 100}


def test_default_reasoning_paths_resolve_to_existing_package():
    from src.utils.config import CONTRADICTION_BENCHMARK_FILE, CLAIM_SUPPORT_BENCHMARK_FILE, HYPOTHESIS_BENCHMARK_FILE
    assert all(Path(p).is_file() for p in [CONTRADICTION_BENCHMARK_FILE,CLAIM_SUPPORT_BENCHMARK_FILE,HYPOTHESIS_BENCHMARK_FILE])
