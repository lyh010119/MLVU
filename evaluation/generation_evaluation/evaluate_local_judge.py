import argparse
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SSC_POINTS = "/NHNHOME/WORKSPACE/0226010268_A/yhlee/MLVU/data/MLVU_videos/test-ground-truth/test_ssc_gt.json"


def load_scoring_points(path=SSC_POINTS):
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {item["question"]: item.get("scoring_points", []) for item in data}


def extract_json(text):
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    for match in reversed(matches):
        try:
            return json.loads(match.replace("'", '"'))
        except Exception:
            continue
    return {}


def clamp_score(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(5.0, value))


def normalize_scores(task, parsed):
    if task == "subplot":
        acc = clamp_score(parsed.get("score_accuracy", 0))
        rel = clamp_score(parsed.get("score_relevance", 0))
        return {
            "score_accuracy": acc,
            "score_relevance": rel,
            "total_score": acc + rel,
        }
    comp = clamp_score(parsed.get("score_completeness", 0))
    reli = clamp_score(parsed.get("score_reliability", 0))
    return {
        "score_completeness": comp,
        "score_reliability": reli,
        "total_score": comp + reli,
    }


def build_prompt(task, sample, scoring_points):
    question = sample["Q"].replace("\n", " ")
    answer = sample["A"]
    pred = sample["pred"]
    if task == "subplot":
        points = scoring_points.get(sample["Q"], [])
        return f"""You are an impartial evaluator for a video sub-scene captioning task.
Score the respondent's answer using two 1-5 scores.

Accuracy:
1 = misses the scoring points.
3 = mentions related content but is partially incorrect or incomplete.
5 = accurately covers the scoring points.

Relevance:
1 = off-topic.
3 = mostly addresses the question but uncertain or incomplete.
5 = fully focused on the question with no irrelevant content.

Question: {question}
Reference answer: {answer}
Scoring points: {points}
Respondent answer: {pred}

Return only one JSON object:
{{"score_accuracy": number, "score_relevance": number, "total_score": number}}"""

    return f"""You are an impartial evaluator for a video summarization task.
Score the respondent's answer using two 1-5 scores.

Completeness:
1 = covers almost none of the main content.
3 = covers most main content but misses important details.
5 = completely covers all key points.

Reliability:
1 = many factual errors or contradictions.
3 = generally accurate with minor errors.
5 = completely accurate, clear, and non-contradictory.

Reference answer: {answer}
Respondent answer: {pred}

Return only one JSON object:
{{"score_completeness": number, "score_reliability": number, "total_score": number}}"""


def generate_judgment(model, tokenizer, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def average(rows, key):
    return sum(row["scores"][key] for row in rows) / len(rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--task", required=True, choices=["subplot", "summary"])
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scoring_points", default=SSC_POINTS)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    with open(args.pred_path, "r") as f:
        samples = json.load(f)
    if args.limit is not None:
        samples = samples[: args.limit]

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    scoring_points = load_scoring_points(args.scoring_points) if args.task == "subplot" else {}
    rows = []
    for sample in tqdm(samples):
        prompt = build_prompt(args.task, sample, scoring_points)
        raw = generate_judgment(model, tokenizer, prompt, args.max_new_tokens)
        scores = normalize_scores(args.task, extract_json(raw))
        rows.append(
            {
                "video_name": sample.get("video_name", ""),
                "scores": scores,
                "raw_judge_output": raw,
            }
        )

    if args.task == "subplot":
        summary = {
            "num_samples": len(rows),
            "score_accuracy": average(rows, "score_accuracy"),
            "score_relevance": average(rows, "score_relevance"),
            "total_score": average(rows, "total_score"),
        }
    else:
        summary = {
            "num_samples": len(rows),
            "score_completeness": average(rows, "score_completeness"),
            "score_reliability": average(rows, "score_reliability"),
            "total_score": average(rows, "total_score"),
        }

    output = {
        "judge_model": args.model_path,
        "task": args.task,
        "summary": summary,
        "per_sample": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
