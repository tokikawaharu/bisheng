"""Unit tests for ``bisheng_langchain.rag.rerank.hybrid_fusion_benchmark``.

The module under test is standard-library only, so it is loaded by file path:
importing it through ``bisheng_langchain.rag`` would pull torch / transformers
through the package ``__init__`` files and make this suite require a GPU image
for what is a pure scoring/ranking check.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "bisheng_langchain"
    / "rag"
    / "rerank"
    / "hybrid_fusion_benchmark.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("hybrid_fusion_benchmark", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_fusion_benchmark"] = module
    spec.loader.exec_module(module)
    return module


hfb = _load_module()


def test_tokenize_keeps_ascii_words_and_cjk_bigrams():
    tokens = hfb.tokenize("RRF 的平滑常数 k 取 60")
    assert "rrf" in tokens
    assert "60" in tokens
    assert "平滑" in tokens
    assert "常数" in tokens


def test_tokenize_drops_particle_led_bigrams():
    assert "的平" not in hfb.tokenize("的平滑常数")


def test_coverage_is_share_of_query_tokens():
    text = "RRF 融合的平滑常数 k 取 60"
    assert hfb.coverage("平滑常数取多少", text) > 0.4
    assert hfb.coverage("今天天气怎么样", text) == 0.0


def test_overlap_count_counts_distinct_tokens():
    text = "切块窗口 512 token，保留 64 token 重叠"
    assert hfb.overlap_count("切块重叠长度", text) >= 2
    assert hfb.overlap_count("股票", text) == 0


def test_bm25_ranks_exact_term_candidate_first():
    texts = [
        "模型网关按指数退避重试，最多三次。",
        "RRF 融合的平滑常数 k 取 60，只看排名。",
        "向量维度 1024，索引类型 IVF_FLAT。",
    ]
    scores = hfb.bm25_scores("RRF 平滑常数", texts)
    assert hfb.rank_by_score(scores)[0] == 1


def test_rrf_fuse_prefers_candidate_ranked_by_both_lists():
    fused = hfb.rrf_fuse([[0, 1, 2], [1, 2, 0]])
    assert fused[0] == 1


def test_rrf_fuse_prefers_candidates_present_in_both_lists():
    """RRF rewards agreement between the two recall paths."""
    fused = hfb.rrf_fuse([[9, 4, 7], [4, 9, 7]])
    assert set(fused[:2]) == {4, 9}
    assert fused[-1] == 7


def test_cross_score_prefers_full_coverage():
    query = "平滑常数 k 取多少"
    good = "RRF 融合的平滑常数 k 取 60。"
    weak = "融合的常数取一个值。"
    assert hfb.cross_score(query, good) > hfb.cross_score(query, weak)


def test_hybrid_rerank_recovers_from_a_bad_dense_top1():
    record = {
        "query": "平滑常数 k 取多少",
        "candidates": [
            {"text": "向量维度 1024，索引类型 IVF_FLAT。", "dense": 0.91},
            {"text": "RRF 融合的平滑常数 k 取 60，只看排名不看分数。", "dense": 0.3},
        ],
        "gold": [1],
    }
    assert hfb.rank_dense(record)[0] == 0
    assert hfb.rank_hybrid_rerank(record)[0] == 1


def test_gate_drops_out_of_domain_queries_only():
    out_of_domain = {
        "query": "今天天气怎么样",
        "candidates": [{"text": "知识库支持 PDF 与 Word 导入。", "dense": 0.5}],
        "gold": [],
    }
    answerable = {
        "query": "知识库支持导入哪些文档格式",
        "candidates": [{"text": "知识库文档支持 PDF、Word、Markdown 与网页链接导入。", "dense": 0.6}],
        "gold": [0],
    }
    assert hfb.rank_full(out_of_domain) == []
    assert hfb.rank_full(answerable)


def test_metrics_match_hand_computed_values():
    ranking = [3, 1, 0, 2]
    gold = [1]
    assert hfb.recall_at_k(ranking, gold, 5) == 1.0
    assert hfb.recall_at_k(ranking, gold, 1) == 0.0
    assert hfb.mrr_at_k(ranking, gold, 5) == pytest.approx(0.5)
    assert hfb.ndcg_at_k(ranking, gold, 5) == pytest.approx(1 / 1.584962500721156)


def test_evaluate_covers_all_strategies_and_bounded_metrics():
    results = hfb.evaluate(hfb.demo_dataset(), k=5)
    assert set(results) == set(hfb.STRATEGIES)
    for metrics in results.values():
        for key in ("recall@5", "mrr@5", "ndcg@5"):
            assert 0.0 <= metrics[key] <= 1.0
    assert results["hybrid_gate"]["gated_queries"] == 2


def test_evaluate_is_deterministic():
    assert hfb.evaluate(hfb.demo_dataset(), k=5) == hfb.evaluate(hfb.demo_dataset(), k=5)


def test_rerank_strategy_beats_raw_fusion_on_demo():
    results = hfb.evaluate(hfb.demo_dataset(), k=1)
    assert results["hybrid_rerank"]["recall@1"] > results["hybrid"]["recall@1"]
    assert results["hybrid_rerank"]["recall@1"] >= results["dense"]["recall@1"]


def test_load_dataset_parses_jsonl(tmp_path):
    path = tmp_path / "eval.jsonl"
    record = {"query": "q", "candidates": [{"text": "t", "dense": 0.5}], "gold": [0]}
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    assert hfb.load_dataset(str(path)) == [record]


def test_load_dataset_rejects_broken_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"query": "q"', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        hfb.load_dataset(str(path))


def test_cli_demo_and_json(capsys):
    assert hfb.main(["--demo"]) == 0
    output = capsys.readouterr().out
    assert "hybrid_rerank" in output
    assert "recall@5" in output

    assert hfb.main(["--demo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["queries"] == len(hfb.demo_dataset())
    assert payload["k"] == 5


def test_cli_requires_a_dataset(capsys):
    with pytest.raises(SystemExit):
        hfb.main([])
