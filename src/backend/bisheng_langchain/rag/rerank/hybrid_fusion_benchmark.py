"""Offline fusion benchmark for BISHENG's dual-recall RAG path.

BISHENG retrieves with Milvus (dense) and Elasticsearch (sparse) and then fuses
the two lists before reranking.  Tuning that fusion currently needs the full
stack up (MySQL + Redis + Milvus + ES) plus a GPU for ``rerank_benchmark.py``.

This module is a service-free companion to ``rerank_benchmark.py``: given a
JSONL file of already-scored candidates it compares four strategies under the
same labels -

    dense      dense score only
    sparse     BM25 (CJK bigram tokenizer, dictionary free)
    hybrid     dense + sparse fused with Reciprocal Rank Fusion
    hybrid_gate hybrid + a coverage gate that drops out-of-domain queries

and reports Recall@k, MRR@k and nDCG@k.  Pure standard library, so it runs in a
laptop shell without the middleware stack and gives reproducible numbers.

Usage::

    python -m bisheng_langchain.rag.rerank.hybrid_fusion_benchmark --demo
    python -m bisheng_langchain.rag.rerank.hybrid_fusion_benchmark --dataset eval.jsonl --k 5

Dataset format, one JSON object per line::

    {"query": "...", "candidates": [{"text": "...", "dense": 0.73}, ...], "gold": [0, 3]}

``gold`` holds candidate indexes that are considered relevant.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from typing import Any, Callable, Sequence

DEFAULT_K = 5
RRF_K = 60
# Out-of-domain gate: the best candidate must share at least this many query
# tokens, and cover at least this share of them.  Both conditions are needed -
# a single long candidate can otherwise look "covered" by two common bigrams.
MIN_OVERLAP = 2
MIN_COVERAGE = 0.10

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-.]*|\d+(?:\.\d+)?")
_PARTICLE_PREFIX = {"的", "了", "是", "着", "呢", "吧", "啊"}


def normalize(text: str) -> str:
    """NFKC-normalise and collapse whitespace so scores are comparable."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().lower()


def tokenize(text: str) -> list[str]:
    """ASCII words plus CJK character bigrams.

    A dictionary-free tokenizer keeps this module dependency-free.  Bigrams
    already carry most of the signal for retrieval on Chinese text; particle-led
    bigrams (``的策``) are dropped as pure noise.
    """
    text = normalize(text)
    tokens: list[str] = _ASCII_WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram[0] in _PARTICLE_PREFIX:
                continue
            tokens.append(bigram)
    return tokens


def coverage(query: str, text: str) -> float:
    """Share of query tokens that appear in ``text``.

    Used as an out-of-domain gate: a question whose own vocabulary is absent
    from the best candidate should be refused instead of answered from it.
    """
    query_tokens = [token for token in tokenize(query) if len(token) > 1]
    if not query_tokens:
        return 0.0
    haystack = normalize(text)
    return sum(1 for token in query_tokens if token in haystack) / len(query_tokens)


def overlap_count(query: str, text: str) -> int:
    """Number of distinct query tokens present in ``text``."""
    haystack = normalize(text)
    return len({token for token in tokenize(query) if len(token) > 1 and token in haystack})


def bm25_scores(query: str, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Score every candidate against ``query`` with BM25."""
    query_tokens = tokenize(query)
    doc_tokens = [tokenize(text) for text in texts]
    doc_len = [max(1, len(tokens)) for tokens in doc_tokens]
    avg_len = sum(doc_len) / max(1, len(doc_len))
    total_docs = len(doc_tokens)

    scores = [0.0] * total_docs
    for token in query_tokens:
        containing = [index for index, tokens in enumerate(doc_tokens) if token in tokens]
        if not containing:
            continue
        idf = math.log(1 + (total_docs - len(containing) + 0.5) / (len(containing) + 0.5))
        for index in containing:
            freq = doc_tokens[index].count(token)
            denom = freq + k1 * (1 - b + b * doc_len[index] / avg_len)
            scores[index] += idf * freq * (k1 + 1) / denom
    return scores


def rank_by_score(scores: Sequence[float], limit: int | None = None) -> list[int]:
    """Return candidate indexes ordered by descending score (stable on ties)."""
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return order if limit is None else order[:limit]


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal Rank Fusion over several ranked index lists.

    RRF consumes ranks rather than raw scores, so a BM25 list and a cosine
    similarity list can be merged without calibrating their scales.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)
    return [index for index, _score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


def _candidate_texts(record: dict[str, Any]) -> list[str]:
    return [str(candidate.get("text", "")) for candidate in record.get("candidates", [])]


def _dense_scores(record: dict[str, Any]) -> list[float]:
    return [float(candidate.get("dense", 0.0)) for candidate in record.get("candidates", [])]


def rank_dense(record: dict[str, Any]) -> list[int]:
    return rank_by_score(_dense_scores(record))


def rank_sparse(record: dict[str, Any]) -> list[int]:
    return rank_by_score(bm25_scores(str(record.get("query", "")), _candidate_texts(record)))


def rank_hybrid(record: dict[str, Any]) -> list[int]:
    return rrf_fuse([rank_dense(record), rank_sparse(record)])


def cross_score(query: str, text: str) -> float:
    """Dependency-free stand-in for a cross-encoder reranker.

    Coverage is raised to a power so that only candidates covering most of the
    query vocabulary reach the top, and a proximity term rewards candidates
    where the matched tokens sit close together.  A real deployment swaps this
    for ``bge-reranker`` (see ``rerank.py``) and keeps the same ordering logic.
    """
    query_tokens = {token for token in tokenize(query) if len(token) > 1}
    if not query_tokens:
        return 0.0
    haystack = normalize(text)
    matched = [token for token in query_tokens if token in haystack]
    if not matched:
        return 0.0
    share = len(matched) / len(query_tokens)
    positions = [haystack.find(token) for token in matched]
    span = max(positions) - min(positions)
    proximity = 1.0 / (1.0 + span / 200.0) if len(positions) > 1 else 0.0
    return share**1.5 + 0.4 * proximity


def rank_hybrid_rerank(record: dict[str, Any], pool: int = 10) -> list[int]:
    """RRF fusion followed by a rerank pass over the fused pool."""
    texts = _candidate_texts(record)
    fused = rank_hybrid(record)[:pool]
    query = str(record.get("query", ""))
    rescored = sorted(fused, key=lambda index: (-cross_score(query, texts[index]), index))
    return rescored + [index for index in rank_hybrid(record) if index not in rescored]


def rank_full(record: dict[str, Any]) -> list[int]:
    """Gate + rerank: the configuration this tool recommends for production."""
    texts = _candidate_texts(record)
    fused = rank_hybrid(record)
    if not fused:
        return []
    query = str(record.get("query", ""))
    best_overlap = max((overlap_count(query, texts[index]) for index in fused[:3]), default=0)
    best_coverage = max((coverage(query, texts[index]) for index in fused[:3]), default=0.0)
    if best_overlap < MIN_OVERLAP or best_coverage < MIN_COVERAGE:
        return []
    return rank_hybrid_rerank(record)


STRATEGIES: dict[str, Callable[[dict[str, Any]], list[int]]] = {
    "dense": rank_dense,
    "sparse": rank_sparse,
    "hybrid": rank_hybrid,
    "hybrid_rerank": rank_hybrid_rerank,
    "hybrid_gate": rank_full,
}


def recall_at_k(ranking: Sequence[int], gold: Sequence[int], k: int) -> float:
    if not gold:
        return 0.0
    hits = len(set(ranking[:k]) & set(gold))
    return hits / len(gold)


def mrr_at_k(ranking: Sequence[int], gold: Sequence[int], k: int) -> float:
    gold_set = set(gold)
    for position, index in enumerate(ranking[:k], start=1):
        if index in gold_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranking: Sequence[int], gold: Sequence[int], k: int) -> float:
    gold_set = set(gold)
    dcg = sum(1.0 / math.log2(position + 1) for position, index in enumerate(ranking[:k], start=1) if index in gold_set)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(gold_set), k) + 1))
    return 0.0 if ideal == 0 else dcg / ideal


def evaluate(dataset: Sequence[dict[str, Any]], k: int = DEFAULT_K) -> dict[str, dict[str, float]]:
    """Score every strategy on the same labelled dataset."""
    results: dict[str, dict[str, float]] = {}
    for name, ranker in STRATEGIES.items():
        recalls, mrrs, ndcgs, gated = [], [], [], 0
        for record in dataset:
            ranking = ranker(record)
            if name == "hybrid_gate" and not ranking:
                gated += 1
            gold = [int(index) for index in record.get("gold", [])]
            recalls.append(recall_at_k(ranking, gold, k))
            mrrs.append(mrr_at_k(ranking, gold, k))
            ndcgs.append(ndcg_at_k(ranking, gold, k))
        count = max(1, len(dataset))
        results[name] = {
            f"recall@{k}": round(sum(recalls) / count, 4),
            f"mrr@{k}": round(sum(mrrs) / count, 4),
            f"ndcg@{k}": round(sum(ndcgs) / count, 4),
            "gated_queries": gated,
        }
    return results


def demo_dataset() -> list[dict[str, Any]]:
    """Small deterministic dataset so the CLI is runnable without any export.

    Dense scores are deliberately adversarial on half of the records (the
    distractor is placed first) which is what a plain embedding model does on
    queries that hinge on a model name, a number or a config key.
    """
    records = [
        {
            "query": "混合检索里 RRF 的平滑常数取多少",
            "candidates": [
                {"text": "RRF 融合的平滑常数 k 取 60，只看排名不看原始分数，避免两路打分器量纲不一致。", "dense": 0.41},
                {"text": "Milvus 负责稠密向量召回，Elasticsearch 负责稀疏召回，两边各取 top 20。", "dense": 0.73},
                {"text": "重排模型使用 bge-reranker-large，阈值 0.3 以上保留。", "dense": 0.44},
                {"text": "知识库文档支持 PDF、Word、Markdown 与网页链接导入。", "dense": 0.39},
                {"text": "租户空间下所有资源按 tenant_id 隔离。", "dense": 0.33},
                {"text": "助手对话支持流式输出与引用溯源。", "dense": 0.3},
            ],
            "gold": [0],
        },
        {
            "query": "文档切块时相邻窗口的重叠长度怎么定",
            "candidates": [
                {"text": "文档先做 OCR 再做版面分析，表格单独抽取。", "dense": 0.61},
                {"text": "切块窗口 512 token，相邻窗口之间保留 64 token 重叠，避免语义被截断。", "dense": 0.72},
                {"text": "向量维度 1024，索引类型 IVF_FLAT，距离度量 COSINE。", "dense": 0.35},
                {"text": "评测数据集可以上传 CSV，逐行生成问答对。", "dense": 0.33},
                {"text": "工作流条件分支用 Jinja2 渲染表达式。", "dense": 0.31},
                {"text": "模型网关按指数退避重试。", "dense": 0.28},
            ],
            "gold": [1],
        },
        {
            "query": "工作流的条件分支怎么配置",
            "candidates": [
                {"text": "工作流条件分支节点按表达式结果选择下游节点，表达式用 Jinja2 渲染。", "dense": 0.66},
                {"text": "知识库支持 PDF、Word、Markdown 与网页链接导入。", "dense": 0.64},
                {"text": "助手对话支持流式输出与引用溯源。", "dense": 0.4},
                {"text": "向量维度 1024。", "dense": 0.36},
                {"text": "租户隔离按 tenant_id。", "dense": 0.32},
                {"text": "评测集支持上传 CSV。", "dense": 0.3},
            ],
            "gold": [0],
        },
        {
            "query": "大模型调用失败后的重试策略怎么设置",
            "candidates": [
                {"text": "评测数据集可以上传 CSV，逐行生成问答对。", "dense": 0.67},
                {"text": "知识库支持 PDF、Word 导入。", "dense": 0.6},
                {"text": "模型网关按指数退避重试，最多三次，超时 60 秒，失败后写失败日志。", "dense": 0.52},
                {"text": "租户空间下所有资源按 tenant_id 隔离。", "dense": 0.34},
                {"text": "向量维度 1024。", "dense": 0.31},
                {"text": "条件分支用 Jinja2。", "dense": 0.29},
            ],
            "gold": [2],
        },
        {
            "query": "知识库支持导入哪些文档格式",
            "candidates": [
                {"text": "知识库文档支持 PDF、Word、Markdown 与网页链接导入，单文件上限 100MB。", "dense": 0.7},
                {"text": "向量维度 1024，索引 IVF_FLAT。", "dense": 0.55},
                {"text": "模型网关按指数退避重试。", "dense": 0.42},
                {"text": "助手对话支持流式输出。", "dense": 0.38},
                {"text": "租户按 tenant_id 隔离。", "dense": 0.33},
                {"text": "评测集支持 CSV。", "dense": 0.3},
            ],
            "gold": [0],
        },
        {
            "query": "助手对话的流式输出与引用怎么实现",
            "candidates": [
                {"text": "条件分支节点用 Jinja2 渲染表达式。", "dense": 0.68},
                {"text": "向量维度 1024。", "dense": 0.57},
                {"text": "知识库支持 PDF 与 Word 导入。", "dense": 0.5},
                {"text": "助手对话支持流式输出与引用溯源，引用按 chunk 编号回传。", "dense": 0.47},
                {"text": "租户按 tenant_id 隔离。", "dense": 0.31},
                {"text": "评测集支持 CSV。", "dense": 0.28},
            ],
            "gold": [3],
        },
        {
            "query": "今天天气怎么样",
            "candidates": [
                {"text": "知识库文档支持 PDF、Word、Markdown 与网页链接导入。", "dense": 0.52},
                {"text": "RRF 融合的平滑常数 k 取 60。", "dense": 0.47},
                {"text": "助手对话支持流式输出与引用溯源。", "dense": 0.4},
                {"text": "向量维度 1024。", "dense": 0.33},
                {"text": "租户按 tenant_id 隔离。", "dense": 0.3},
                {"text": "评测集支持 CSV。", "dense": 0.27},
            ],
            "gold": [],
        },
        {
            "query": "推荐一只股票",
            "candidates": [
                {"text": "模型网关按指数退避重试，最多三次。", "dense": 0.55},
                {"text": "向量维度 1024，索引类型 IVF_FLAT。", "dense": 0.5},
                {"text": "知识库支持 PDF、Word、Markdown 与网页链接导入。", "dense": 0.38},
                {"text": "条件分支用 Jinja2 渲染。", "dense": 0.32},
                {"text": "助手对话支持流式输出。", "dense": 0.29},
                {"text": "租户按 tenant_id 隔离。", "dense": 0.26},
            ],
            "gold": [],
        },
    ]
    return records


def load_dataset(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {error}") from error
    return records


def format_table(results: dict[str, dict[str, float]], k: int) -> str:
    header = f"{'strategy':<12}{'recall@' + str(k):>12}{'mrr@' + str(k):>10}{'ndcg@' + str(k):>10}{'gated':>8}"
    lines = [header, "-" * len(header)]
    for name, metrics in results.items():
        lines.append(
            f"{name:<12}{metrics[f'recall@{k}']:>12.4f}{metrics[f'mrr@{k}']:>10.4f}"
            f"{metrics[f'ndcg@{k}']:>10.4f}{int(metrics['gated_queries']):>8d}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline fusion benchmark for BISHENG dual-recall RAG")
    parser.add_argument("--dataset", help="JSONL file of scored candidates (see module docstring)")
    parser.add_argument("--demo", action="store_true", help="run on the built-in demo dataset")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="cut-off for recall/MRR/nDCG")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    if args.dataset and args.demo:
        parser.error("choose either --dataset or --demo")
    if args.dataset:
        dataset = load_dataset(args.dataset)
    elif args.demo:
        dataset = demo_dataset()
    else:
        parser.error("one of --dataset or --demo is required")
    if not dataset:
        print("empty dataset", file=sys.stderr)
        return 2

    results = evaluate(dataset, k=args.k)
    if args.json:
        print(json.dumps({"queries": len(dataset), "k": args.k, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(f"queries: {len(dataset)}  k: {args.k}")
        print(format_table(results, args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
