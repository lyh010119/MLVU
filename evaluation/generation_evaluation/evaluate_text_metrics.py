import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import sacrebleu


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", normalize(text), flags=re.UNICODE)


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def f1_from_counts(pred_tokens, ref_tokens):
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred_tokens, ref_tokens):
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    beta = precision / (recall + 1e-12)
    return ((1 + beta * beta) * precision * recall) / (recall + beta * beta * precision + 1e-12)


def score_sample(sample):
    pred = normalize(sample.get("pred", ""))
    ref = normalize(sample.get("A", ""))
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)

    bleu = sacrebleu.sentence_bleu(pred, [ref]).score
    chrf = sacrebleu.sentence_chrf(pred, [ref]).score
    return {
        "video_name": sample.get("video_name", ""),
        "token_f1": f1_from_counts(pred_tokens, ref_tokens) * 100,
        "rouge_l": rouge_l(pred_tokens, ref_tokens) * 100,
        "bleu": bleu,
        "chrf": chrf,
        "pred_len": len(pred_tokens),
        "ref_len": len(ref_tokens),
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    with open(args.pred_path, "r") as f:
        samples = json.load(f)

    per_sample = [score_sample(sample) for sample in samples]
    summary = {
        "num_samples": len(per_sample),
        "token_f1": mean(x["token_f1"] for x in per_sample),
        "rouge_l": mean(x["rouge_l"] for x in per_sample),
        "bleu": mean(x["bleu"] for x in per_sample),
        "chrf": mean(x["chrf"] for x in per_sample),
        "pred_len": mean(x["pred_len"] for x in per_sample),
        "ref_len": mean(x["ref_len"] for x in per_sample),
    }

    output = {"summary": summary, "per_sample": per_sample}
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
