import argparse
import json
import os
import re
from collections import Counter, defaultdict
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
        return "".join(ans).strip()
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

def save_checkpoint(output_file, pass1_correct, pass16_correct, maj16_correct, total, category_stats, level_stats, records):
    acc_pass1 = (pass1_correct / total * 100) if total > 0 else 0.0
    acc_pass16 = (pass16_correct / total * 100) if total > 0 else 0.0
    acc_maj16 = (maj16_correct / total * 100) if total > 0 else 0.0

    temp_file = output_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump({
            "total_problems": total,
            "pass@1": round(acc_pass1, 2),
            "pass@16": round(acc_pass16, 2),
            "maj@16": round(acc_maj16, 2),
            "category_stats": category_stats,
            "level_stats": level_stats,
            "details": records
        }, f, indent=2)
    os.replace(temp_file, output_file)

def get_balanced_subset(dataset, samples_per_category=20):
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
    parser.add_argument("--samples_per_category", type=int, default=50, help="Samples per category (50 -> 350 total, 20 -> 140 total)")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of rollouts per problem (e.g. 16 for Pass@16)")
    parser.add_argument("--chunk_size", type=int, default=4, help="Batch generation chunk size to avoid GPU OOM")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--output_file", type=str, default="results_qwen3_base_pass16.json")
    args = parser.parse_args()

    print("=" * 60)
    print("QWEN3-1.7B BASE MODEL EVALUATION (PASS@16 / MAJ@16)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Rollouts per problem: {args.num_samples} (in chunks of {args.chunk_size})")
    print(f"Temperature: {args.temperature} | Top-p: {args.top_p}")
    print(f"Max New Tokens: {args.max_new_tokens}")

    model_path = args.model_path
    if not os.path.exists(model_path):
        if os.path.exists("./Qwen3-1.7B"):
            model_path = "./Qwen3-1.7B"
        elif os.path.exists("/scratch/arushishukla/anlp_project/Qwen3-1.7B"):
            model_path = "/scratch/arushishukla/anlp_project/Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=os.path.isdir(model_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path)
    )
    model.eval()

    print(f"\nLoading dataset 'qwedsacf/competition_math' (split: {args.split})...")
    try:
        dataset = load_dataset("qwedsacf/competition_math", split=args.split)
    except Exception:
        if os.path.exists("./competition_math"):
            dataset = load_dataset("./competition_math", split=args.split)
        elif os.path.exists("/scratch/arushishukla/anlp_project/competition_math"):
            dataset = load_dataset("/scratch/arushishukla/anlp_project/competition_math", split=args.split)
        else:
            raise
    eval_dataset = get_balanced_subset(dataset, samples_per_category=args.samples_per_category)

    pass1_correct = 0
    pass16_correct = 0
    maj16_correct = 0
    total = 0

    category_stats = {}
    level_stats = {}
    records = []

    num_chunks = max(1, args.num_samples // args.chunk_size)

    for item in tqdm(eval_dataset, desc="Evaluating Problems"):
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

        sampled_predictions = []
        is_any_correct = False
        first_sample_correct = False

        # Generate in chunks of 4 to prevent CUDA OOM
        for chunk_idx in range(num_chunks):
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.chunk_size,
                    pad_token_id=tokenizer.pad_token_id
                )

            for s_idx, seq in enumerate(generated_ids):
                global_idx = chunk_idx * args.chunk_size + s_idx
                output_text = tokenizer.decode(seq[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                pred_answer = extract_boxed_answer(output_text)
                sampled_predictions.append(pred_answer)

                correct = is_correct(pred_answer, ground_truth)
                if global_idx == 0 and correct:
                    first_sample_correct = True
                if correct:
                    is_any_correct = True

        if first_sample_correct:
            pass1_correct += 1

        # PASS@16: If AT LEAST 1 of the 16 outputs is correct -> Success!
        if is_any_correct:
            pass16_correct += 1

        # Majority Vote among valid predictions
        valid_answers = [ans for ans in sampled_predictions if ans]
        maj_correct = False
        if valid_answers:
            most_common_answer = Counter(valid_answers).most_common(1)[0][0]
            if is_correct(most_common_answer, ground_truth):
                maj16_correct += 1
                maj_correct = True

        total += 1

        category_stats.setdefault(prob_type, {"pass1": 0, "pass16": 0, "maj16": 0, "total": 0})
        category_stats[prob_type]["total"] += 1
        if first_sample_correct:
            category_stats[prob_type]["pass1"] += 1
        if is_any_correct:
            category_stats[prob_type]["pass16"] += 1
        if maj_correct:
            category_stats[prob_type]["maj16"] += 1

        level_stats.setdefault(prob_level, {"pass1": 0, "pass16": 0, "maj16": 0, "total": 0})
        level_stats[prob_level]["total"] += 1
        if first_sample_correct:
            level_stats[prob_level]["pass1"] += 1
        if is_any_correct:
            level_stats[prob_level]["pass16"] += 1
        if maj_correct:
            level_stats[prob_level]["maj16"] += 1

        records.append({
            "problem": problem,
            "type": prob_type,
            "level": prob_level,
            "ground_truth": ground_truth,
            "pass1_correct": first_sample_correct,
            "pass16_correct": is_any_correct,
            "maj16_correct": maj_correct,
            "sampled_predictions": sampled_predictions
        })

        # Save checkpoint after every problem
        save_checkpoint(args.output_file, pass1_correct, pass16_correct, maj16_correct, total, category_stats, level_stats, records)

        if total % 5 == 0 or total == len(eval_dataset):
            p1 = (pass1_correct / total) * 100
            p16 = (pass16_correct / total) * 100
            m16 = (maj16_correct / total) * 100
            print(f"\n[Progress {total}/{len(eval_dataset)}] Pass@1: {p1:.1f}% | Pass@16: {p16:.1f}% | Maj@16: {m16:.1f}%", flush=True)

    final_p1 = (pass1_correct / total) * 100
    final_p16 = (pass16_correct / total) * 100
    final_m16 = (maj16_correct / total) * 100

    print("\n" + "=" * 60)
    print("                     FINAL RESULTS")
    print("=" * 60)
    print(f"Total Problems:   {total}")
    print(f"Pass@1 Accuracy:  {final_p1:.2f}% ({pass1_correct}/{total})")
    print(f"Pass@16 Accuracy: {final_p16:.2f}% ({pass16_correct}/{total})")
    print(f"Maj@16 Accuracy:  {final_m16:.2f}% ({maj16_correct}/{total})")
    print("=" * 60)

if __name__ == "__main__":
    main()
