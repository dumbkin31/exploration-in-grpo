import argparse
import json
import os
import re
from collections import defaultdict
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def extract_boxed_answer(text: str):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    idx += len("\\boxed{")
    brace_count = 1
    ans = []
    for char in text[idx:]:
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                break
        ans.append(char)
    if brace_count == 0:
        extracted = "".join(ans).strip()
        return extracted
    return None

def normalize_math_answer(ans: str):
    if ans is None:
        return ""
    ans = ans.strip()
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", ans)
    ans = ans.replace("\\$", "").replace("$", "").replace(" ", "").replace(",", "")
    ans = ans.replace("\\%", "").replace("%", "")
    ans = re.sub(r"\\dfrac", r"\\frac", ans)
    return ans

def is_correct(pred_str, gt_str):
    pred_clean = normalize_math_answer(pred_str)
    gt_clean = normalize_math_answer(gt_str)
    if not pred_clean or not gt_clean:
        return False
    return pred_clean == gt_clean

def save_checkpoint(output_file, correct, total, category_stats, level_stats, records):
    acc = (correct / total * 100) if total > 0 else 0.0
    temp_file = output_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump({
            "accuracy": acc,
            "total": total,
            "correct": correct,
            "category_stats": category_stats,
            "level_stats": level_stats,
            "details": records
        }, f, indent=2)
    os.replace(temp_file, output_file)

def get_balanced_subset(dataset, samples_per_category=50, seed=42):
    """
    Splits dataset into categories and takes an exact balanced number from each.
    """
    categorized = defaultdict(list)
    for item in dataset:
        cat = item.get("type", "Unknown")
        categorized[cat].append(item)

    balanced_list = []
    print("\n--- Category Breakdown ---")
    for cat, items in categorized.items():
        selected = items[:samples_per_category]
        balanced_list.extend(selected)
        print(f"  • {cat:<25}: selected {len(selected)} / {len(items)} available")
    print(f"Total balanced evaluation samples: {len(balanced_list)}\n")
    return balanced_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--samples_per_category", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="results_qwen3_base_stratified_350.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"Loading dataset 'qwedsacf/competition_math' (split: {args.split})...")
    dataset = load_dataset("qwedsacf/competition_math", split=args.split)

    # Pick exactly 50 from each category
    eval_dataset = get_balanced_subset(dataset, samples_per_category=args.samples_per_category)

    correct = 0
    total = 0
    category_stats = {}
    level_stats = {}
    records = []

    print(f"Evaluating {len(eval_dataset)} samples in NON-THINKING mode with continuous autosave...")

    for i, item in enumerate(tqdm(eval_dataset)):
        problem = item["problem"]
        gt_solution = item["solution"]
        ground_truth = extract_boxed_answer(gt_solution)
        prob_type = item.get("type", "Unknown")
        prob_level = item.get("level", "Unknown")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem}
        ]

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        pred_answer = extract_boxed_answer(output_text)
        match = is_correct(pred_answer, ground_truth)

        if match:
            correct += 1
        total += 1

        category_stats.setdefault(prob_type, {"correct": 0, "total": 0})
        category_stats[prob_type]["total"] += 1
        if match:
            category_stats[prob_type]["correct"] += 1

        level_stats.setdefault(prob_level, {"correct": 0, "total": 0})
        level_stats[prob_level]["total"] += 1
        if match:
            level_stats[prob_level]["correct"] += 1

        records.append({
            "problem": problem,
            "type": prob_type,
            "level": prob_level,
            "ground_truth": ground_truth,
            "prediction_boxed": pred_answer,
            "correct": match,
            "output_text": output_text
        })

        # Save checkpoint after every single sample
        save_checkpoint(args.output_file, correct, total, category_stats, level_stats, records)

        # Print running status every 10 samples
        if total % 10 == 0 or total == len(eval_dataset):
            current_acc = (correct / total) * 100
            print(f"\n[Progress {total}/{len(eval_dataset)}] Running Pass@1: {current_acc:.2f}% ({correct}/{total})", flush=True)

    final_acc = (correct / total) * 100
    print("\n" + "=" * 60)
    print(f"FINAL STRATIFIED PASS@1 ACCURACY: {final_acc:.2f}% ({correct}/{total})")
    print("=" * 60)
    print(json.dumps(category_stats, indent=2))

if __name__ == "__main__":
    main()
