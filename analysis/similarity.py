import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


def get_totals(rollout_list):
    total_std = 0
    total_cuts = 0
    for rollout in rollout_list:
        if rollout["rollout_kind"] == "std":
            total_std += 1
        else:
            total_cuts += 1

    return total_cuts, total_std


def correctness(rollout_list):
    correct_std = 0
    correct_cuts = 0
    for rollout in rollout_list:
        if rollout["rollout_kind"] == "std":
            correct_std += rollout["score"]
        else:
            correct_cuts += rollout["score"]
    return correct_cuts, correct_std


def validity(rollout_list):
    valid_std = 0
    valid_cuts = 0
    for rollout in rollout_list:
        if r"\boxed{" not in rollout["output"]:
            continue
        if rollout["rollout_kind"] == "std":
            valid_std += 1
        else:
            valid_cuts += 1
    return valid_cuts, valid_std


def length_buckets(rollout_list):
    buckets = defaultdict(list)
    for rollout in rollout_list:
        buckets[len(rollout["output"])].append(rollout)
    return buckets


def _prompt_id(rollout):
    return rollout["uid"].split("_")[0]


def _rollouts_by_prompt_and_kind(rollout_list):
    grouped_rollouts = defaultdict(lambda: {"cuts": [], "std": []})
    for rollout in rollout_list:
        rollout_kind = rollout["rollout_kind"]
        if rollout_kind in ("cuts", "std"):
            grouped_rollouts[(rollout["step"], _prompt_id(rollout))][rollout_kind].append(rollout)
    return grouped_rollouts


def compare_rollout_outputs(
    rollout_list,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_model=None,
):
    """Compare every CUTS output with every STD output for the same prompt."""
    grouped_rollouts = _rollouts_by_prompt_and_kind(rollout_list)
    if embedding_model is None:
        embedding_model = SentenceTransformer(embedding_model_name)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    comparisons = []

    for (step, prompt_id), prompt_rollouts in grouped_rollouts.items():
        outputs = [
            rollout["output"]
            for rollout in prompt_rollouts["cuts"] + prompt_rollouts["std"]
        ]
        if not outputs:
            continue
        embeddings = embedding_model.encode(outputs, convert_to_numpy=True, normalize_embeddings=True)
        cuts_count = len(prompt_rollouts["cuts"])
        cuts_embeddings = embeddings[:cuts_count]
        std_embeddings = embeddings[cuts_count:]

        for cuts_rollout, cuts_embedding in zip(prompt_rollouts["cuts"], cuts_embeddings):
            for std_rollout, std_embedding in zip(prompt_rollouts["std"], std_embeddings):
                overlap = scorer.score(cuts_rollout["output"], std_rollout["output"])
                comparisons.append(
                    {
                        "prompt_id": prompt_id,
                        "step": step,
                        "cuts_uid": cuts_rollout["uid"],
                        "std_uid": std_rollout["uid"],
                        "cosine_similarity": float(util.cos_sim(cuts_embedding, std_embedding).item()),
                        "rouge1_f1": overlap["rouge1"].fmeasure,
                        "rouge2_f1": overlap["rouge2"].fmeasure,
                        "rougeL_f1": overlap["rougeL"].fmeasure,
                    }
                )
    return comparisons


def compare_std_rollouts(
    rollout_list,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    embedding_model=None,
):
    """Compare every pair of STD outputs for each prompt and training step."""
    grouped_rollouts = _rollouts_by_prompt_and_kind(rollout_list)
    if embedding_model is None:
        embedding_model = SentenceTransformer(embedding_model_name)
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    comparisons = []

    for (step, prompt_id), prompt_rollouts in grouped_rollouts.items():
        std_rollouts = prompt_rollouts["std"]
        if len(std_rollouts) < 2:
            continue
        outputs = [rollout["output"] for rollout in std_rollouts]
        embeddings = embedding_model.encode(
            outputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for (first_rollout, first_embedding), (second_rollout, second_embedding) in combinations(
            zip(std_rollouts, embeddings),
            2,
        ):
            overlap = scorer.score(first_rollout["output"], second_rollout["output"])
            comparisons.append(
                {
                    "prompt_id": prompt_id,
                    "step": step,
                    "first_std_uid": first_rollout["uid"],
                    "second_std_uid": second_rollout["uid"],
                    "cosine_similarity": float(
                        util.cos_sim(first_embedding, second_embedding).item()
                    ),
                    "ngram_overlap": overlap["rouge1"].fmeasure,
                }
            )
    return comparisons


def summarize_comparisons(comparisons):
    """Return metric averages for each training step and across all steps."""
    metric_names = ("cosine_similarity", "rouge1_f1", "rouge2_f1", "rougeL_f1")
    metrics_by_step = defaultdict(lambda: defaultdict(list))
    for comparison in comparisons:
        for metric_name in metric_names:
            metrics_by_step[comparison["step"]][metric_name].append(comparison[metric_name])

    def averages(metric_values):
        return {
            metric_name: float(np.mean(values))
            for metric_name, values in metric_values.items()
        }

    per_step = {
        str(step): averages(metric_values)
        for step, metric_values in sorted(metrics_by_step.items())
    }
    all_values = defaultdict(list)
    for metric_values in metrics_by_step.values():
        for metric_name, values in metric_values.items():
            all_values[metric_name].extend(values)
    return {"by_step": per_step, "overall": averages(all_values)}


def summarize_std_comparisons(comparisons):
    """Summarize STD self-similarity per prompt, training step, and overall."""
    metric_names = ("cosine_similarity", "ngram_overlap")
    metrics_by_step = defaultdict(lambda: defaultdict(list))
    metrics_by_prompt = defaultdict(lambda: defaultdict(list))
    for comparison in comparisons:
        for metric_name in metric_names:
            value = comparison[metric_name]
            metrics_by_step[comparison["step"]][metric_name].append(value)
            metrics_by_prompt[comparison["prompt_id"]][metric_name].append(value)

    def averages(metric_values):
        return {
            metric_name: float(np.mean(values))
            for metric_name, values in metric_values.items()
            if values
        }

    return {
        "by_prompt": {
            prompt_id: averages(metric_values)
            for prompt_id, metric_values in sorted(metrics_by_prompt.items())
        },
        "by_step": {
            str(step): averages(metric_values)
            for step, metric_values in sorted(metrics_by_step.items())
        },
        "overall": averages(
            {
                metric_name: [
                    value
                    for metric_values in metrics_by_step.values()
                    for value in metric_values[metric_name]
                ]
                for metric_name in metric_names
            }
        ),
    }


def load_rollout_list(rollout_dumps_dir):
    """Load every numeric-step JSONL dump into one rollout list."""
    rollout_dumps_path = Path(rollout_dumps_dir)
    rollout_list = []
    for dump_path in sorted(
        rollout_dumps_path.glob("*.jsonl"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else float("inf"),
    ):
        if not dump_path.stem.isdigit():
            continue
        step = int(dump_path.stem)
        with dump_path.open() as dump_file:
            for line in dump_file:
                if not line.strip():
                    continue
                rollout = json.loads(line)
                rollout.setdefault("step", step)
                rollout_list.append(rollout)
    return rollout_list


def output_length_score_correlation(rollout_list):
    """Return Pearson and Spearman correlations between character count and score."""
    output_lengths = np.array([len(rollout["output"]) for rollout in rollout_list])
    scores = np.array([rollout["score"] for rollout in rollout_list])
    if len(output_lengths) < 2:
        return {"pearson": None, "spearman": None}
    pearson = pd.Series(output_lengths).corr(pd.Series(scores), method="pearson")
    spearman = pd.Series(output_lengths).corr(pd.Series(scores), method="spearman")
    return {
        "pearson": float(pearson) if pd.notna(pearson) else None,
        "spearman": float(spearman) if pd.notna(spearman) else None,
    }


def _plot_rollout_count_by_length(rollout_list, count_name, output_path):
    counts = defaultdict(lambda: {"cuts": 0, "std": 0})
    for rollout in rollout_list:
        output_length = len(rollout["output"])
        rollout_kind = rollout["rollout_kind"]
        if rollout_kind not in ("cuts", "std"):
            continue
        if count_name == "correct" and rollout["score"]:
            counts[output_length][rollout_kind] += 1
        elif count_name == "valid" and r"\boxed{" in rollout["output"]:
            counts[output_length][rollout_kind] += 1

    output_lengths = sorted(counts)
    figure, axis = plt.subplots()
    for rollout_kind in ("cuts", "std"):
        axis.plot(
            output_lengths,
            [counts[length][rollout_kind] for length in output_lengths],
            marker="o",
            label=rollout_kind,
        )
    axis.set_xlabel("output length (characters)")
    axis.set_ylabel(f"number of {count_name} rollouts")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return figure, axis


def plot_correct_rollouts_by_length(rollout_list, output_dir="analysis/results"):
    output_path = Path(output_dir) / "correct_rollouts_by_length.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _plot_rollout_count_by_length(rollout_list, "correct", output_path)


def plot_valid_rollouts_by_length(rollout_list, output_dir="analysis/results"):
    output_path = Path(output_dir) / "valid_rollouts_by_length.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _plot_rollout_count_by_length(rollout_list, "valid", output_path)


def save_summary(summary, output_path="analysis/results/summary.json"):
    """Save all analysis results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def analyze_rollouts(rollout_list, output_dir="analysis/results"):
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    comparisons = compare_rollout_outputs(rollout_list, embedding_model=embedding_model)
    std_comparisons = compare_std_rollouts(rollout_list, embedding_model=embedding_model)
    summary = {
        "totals": get_totals(rollout_list),
        "correctness": correctness(rollout_list),
        "validity": validity(rollout_list),
        "comparison": summarize_comparisons(comparisons),
        "std_self_similarity": summarize_std_comparisons(std_comparisons),
        "output_length_score_correlation": output_length_score_correlation(rollout_list),
    }
    plot_correct_rollouts_by_length(rollout_list, output_dir)
    plot_valid_rollouts_by_length(rollout_list, output_dir)
    save_summary(summary, Path(output_dir) / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dumps_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    arguments = parser.parse_args()
    rollout_list = load_rollout_list(arguments.rollout_dumps_dir)
    analyze_rollouts(rollout_list, arguments.output_dir)


if __name__ == "__main__":
    main()