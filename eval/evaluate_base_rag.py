import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_service import BaseRagSummarizeService, HybridRagSummarizeService
from utils.path_tool import get_abs_path

DEFAULT_K_VALUES = [1, 3, 5, 10]
SUPPORTED_SCHEMES = {"A", "C"}


@dataclass
class MatchResult:
    source_hit_rank: int | None
    strict_hit_rank: int | None


def normalize_text(text: str) -> str:
    return "".join(text.lower().split())


def normalize_source(source: str) -> str:
    return Path(source).name.lower()


def load_dataset(dataset_path: str) -> list[dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def keyword_hit_count(doc_text: str, keywords: list[str]) -> int:
    normalized_doc = normalize_text(doc_text)
    return sum(1 for keyword in keywords if normalize_text(keyword) in normalized_doc)


def source_matches(doc_source: str, target_source: str) -> bool:
    return normalize_source(doc_source) == normalize_source(target_source)


def doc_matches_target(doc: Any, target: dict[str, Any]) -> tuple[bool, bool, int]:
    doc_source = doc.metadata.get("source", "")
    if not source_matches(doc_source, target["source"]):
        return False, False, 0

    keywords = target.get("keywords", [])
    min_keyword_hits = target.get("min_keyword_hits", 0 if not keywords else len(keywords))
    hits = keyword_hit_count(doc.page_content, keywords)
    strict_match = hits >= min_keyword_hits
    return True, strict_match, hits


def find_hit_ranks(docs: list[Any], case: dict[str, Any]) -> MatchResult:
    source_hit_rank = None
    strict_hit_rank = None

    for index, doc in enumerate(docs, start=1):
        for target in case["gold_targets"]:
            source_match, strict_match, _ = doc_matches_target(doc, target)
            if source_match and source_hit_rank is None:
                source_hit_rank = index
            if strict_match and strict_hit_rank is None:
                strict_hit_rank = index
        if source_hit_rank is not None and strict_hit_rank is not None:
            break

    return MatchResult(source_hit_rank=source_hit_rank, strict_hit_rank=strict_hit_rank)


def build_case_detail(case: dict[str, Any], docs: list[Any], hit_ranks: MatchResult) -> dict[str, Any]:
    retrieved = []
    for rank, doc in enumerate(docs, start=1):
        retrieved.append(
            {
                "rank": rank,
                "source": doc.metadata.get("source", ""),
                "preview": doc.page_content[:120].replace("\n", " "),
                "vector_rank": doc.metadata.get("vector_rank"),
                "bm25_rank": doc.metadata.get("bm25_rank"),
                "hybrid_rank": doc.metadata.get("hybrid_rank"),
                "rerank_score": doc.metadata.get("rerank_score"),
            }
        )

    return {
        "id": case["id"],
        "query": case["query"],
        "source_hit_rank": hit_ranks.source_hit_rank,
        "strict_hit_rank": hit_ranks.strict_hit_rank,
        "gold_targets": case["gold_targets"],
        "retrieved_docs": retrieved,
    }


def recall_at_k(ranks: list[int | None], k: int) -> float:
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks) if ranks else 0.0


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    values = [1 / rank for rank in ranks if rank is not None]
    return statistics.fmean(values) if values else 0.0


def make_metrics(ranks: list[int | None], k_values: list[int]) -> dict[str, float]:
    metrics = {f"Recall@{k}": recall_at_k(ranks, k) for k in k_values}
    metrics["MRR"] = mean_reciprocal_rank(ranks)
    return metrics


def build_retriever(scheme: str):
    scheme = scheme.upper()
    if scheme == "A":
        return BaseRagSummarizeService()
    if scheme == "C":
        return HybridRagSummarizeService()
    raise ValueError(f"Unsupported scheme: {scheme}")


def evaluate(dataset_path: str, k_values: list[int], scheme: str) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    max_k = max(k_values)
    service = build_retriever(scheme)

    details = []
    source_ranks = []
    strict_ranks = []

    for case in dataset:
        docs = service.retriever_docs(case["query"], top_k=max_k)
        hit_ranks = find_hit_ranks(docs, case)
        details.append(build_case_detail(case, docs, hit_ranks))
        source_ranks.append(hit_ranks.source_hit_rank)
        strict_ranks.append(hit_ranks.strict_hit_rank)

    return {
        "scheme": scheme,
        "dataset_path": dataset_path,
        "case_count": len(dataset),
        "k_values": k_values,
        "source_metrics": make_metrics(source_ranks, k_values),
        "strict_metrics": make_metrics(strict_ranks, k_values),
        "details": details,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"=== RAG Retrieval Evaluation: Scheme {summary['scheme']} ===")
    print(f"Dataset: {summary['dataset_path']}")
    print(f"Cases:   {summary['case_count']}")
    print(f"K vals:  {summary['k_values']}")
    print()
    print("Source-level recall")
    for key, value in summary["source_metrics"].items():
        print(f"  {key}: {value:.4f}")
    print()
    print("Strict recall (source + keyword snippets)")
    for key, value in summary["strict_metrics"].items():
        print(f"  {key}: {value:.4f}")
    print()
    print("Per-case ranks")
    for detail in summary["details"]:
        print(
            f"  {detail['id']}: source_hit_rank={detail['source_hit_rank']}, "
            f"strict_hit_rank={detail['strict_hit_rank']} | {detail['query']}"
        )


def print_comparison(left: dict[str, Any], right: dict[str, Any]) -> None:
    print("=== Scheme Comparison ===")
    print(f"Dataset: {left['dataset_path']}")
    print(f"Cases:   {left['case_count']}")
    print()
    print(f"{'Metric':<18}{'A':>10}{'C':>10}{'Delta(C-A)':>14}")
    for metric_name in left["source_metrics"]:
        a_value = left["source_metrics"][metric_name]
        c_value = right["source_metrics"][metric_name]
        print(f"{'Source ' + metric_name:<18}{a_value:>10.4f}{c_value:>10.4f}{(c_value - a_value):>14.4f}")
    for metric_name in left["strict_metrics"]:
        a_value = left["strict_metrics"][metric_name]
        c_value = right["strict_metrics"][metric_name]
        print(f"{'Strict ' + metric_name:<18}{a_value:>10.4f}{c_value:>10.4f}{(c_value - a_value):>14.4f}")


def save_summary(summary: dict[str, Any], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"rag_eval_scheme_{summary['scheme'].lower()}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return str(output_path)


def save_comparison(comparison: dict[str, Any], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"rag_eval_comparison_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval recall for scheme A and scheme C.")
    parser.add_argument(
        "--dataset",
        default=get_abs_path("eval/base_rag_recall_dataset.json"),
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="Recall@k values to compute.",
    )
    parser.add_argument(
        "--scheme",
        default="A",
        choices=["A", "C", "both"],
        help="Which scheme to evaluate. Use 'both' to run A/C comparison together.",
    )
    parser.add_argument(
        "--output-dir",
        default=get_abs_path("eval/results"),
        help="Directory to save detailed evaluation results.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the evaluation summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scheme in SUPPORTED_SCHEMES:
        summary = evaluate(args.dataset, args.k_values, args.scheme)
        print_summary(summary)

        if not args.no_save:
            output_path = save_summary(summary, args.output_dir)
            print()
            print(f"Saved detailed results to: {output_path}")
        return

    summary_a = evaluate(args.dataset, args.k_values, "A")
    summary_c = evaluate(args.dataset, args.k_values, "C")

    print_summary(summary_a)
    print()
    print_summary(summary_c)
    print()
    print_comparison(summary_a, summary_c)

    if not args.no_save:
        output_path = save_comparison(
            {
                "dataset_path": args.dataset,
                "k_values": args.k_values,
                "summaries": {"A": summary_a, "C": summary_c},
            },
            args.output_dir,
        )
        print()
        print(f"Saved comparison results to: {output_path}")


if __name__ == "__main__":
    main()
