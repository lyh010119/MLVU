import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from tqdm import tqdm


LONGVA_ROOT = Path("/NHNHOME/WORKSPACE/0226010268_A/yhlee/LongVA")
if str(LONGVA_ROOT) not in sys.path:
    sys.path.append(str(LONGVA_ROOT))

from longva.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from longva.conversation import conv_templates
from longva.mm_utils import tokenizer_image_token
from longva.model.builder import load_pretrained_model


def load_video(video_path, num_frames=256):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"empty video: {video_path}")
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return vr.get_batch(frame_indices).asnumpy()


def get_prompt2(conv):
    ret = conv.system + conv.sep
    for idx, (role, message) in enumerate(conv.messages):
        is_last = idx == len(conv.messages) - 1
        if is_last:
            ret += role + ": " + (message if message is not None else "")
        elif message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


class MLVUGeneration(Dataset):
    def __init__(self, gt_dir, video_dir, tasks, limit=None):
        self.data_list = []
        task_specs = {
            "subPlot": "test_ssc_gt.json",
            "summary": "test_vs_gt.json",
        }

        for task in tasks:
            json_path = Path(gt_dir) / task_specs[task]
            with open(json_path, "r") as f:
                json_data = json.load(f)
            if limit is not None:
                json_data = json_data[:limit]

            for item in json_data:
                video_name = item["video"]
                video_path = Path(video_dir) / video_name
                if not video_path.exists():
                    raise FileNotFoundError(f"video not found: {video_path}")
                self.data_list.append(
                    {
                        "task_type": task,
                        "video_name": video_name,
                        "video": video_path,
                        "question": item["question"],
                        "answer": item["answer"],
                    }
                )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def set_kv_config(model):
    model.config.kv_mode = int(os.getenv("KV_MODE", 1))
    model.config.kv_budget = int(os.getenv("KV_BUDGET", 0))
    model.config.kv_window = int(os.getenv("KV_WINDOW", 32))
    model.config.kv_sink = int(os.getenv("KV_SINK", 4))
    model.config.kv_alpha = float(os.getenv("KV_ALPHA", 0.5))
    model.config.kv_tau = float(os.getenv("KV_TAU", 3.0))
    model.config.kv_gamma = float(os.getenv("KV_GAMMA", 0.9))
    model.config.kv_kernel_size = int(os.getenv("KV_KERNEL_SIZE", 7))


def build_prompt(question):
    conv = conv_templates["qwen_1_5"].copy()
    conv.system = (
        "Carefully watch this video and pay attention to every detail. "
        "Based on your observations, answer the given questions. "
        "Answer in English only."
    )
    
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return get_prompt2(conv)


def run_generation(model, tokenizer, image_processor, example, num_frames, max_new_tokens):
    video_np = load_video(example["video"], num_frames=num_frames)
    video_tensor = image_processor.preprocess(video_np, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(model.device, dtype=torch.float16)

    prompt = build_prompt(example["question"])
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            modalities=["video"],
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir",
        default="/NHNHOME/WORKSPACE/0226010268_A/yhlee/MLVU/data/MLVU_videos/test-ground-truth",
    )
    parser.add_argument(
        "--video_dir",
        default="/NHNHOME/WORKSPACE/0226010268_A/yhlee/MLVU/data/MLVU_videos/MLVU_Test/video",
    )
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--model_path", default="lmms-lab/LongVA-7B")
    parser.add_argument("--num_frames", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--tasks", default="subPlot,summary")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    for task in tasks:
        if task not in {"subPlot", "summary"}:
            raise ValueError(f"unknown task: {task}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MLVUGeneration(args.gt_dir, args.video_dir, tasks, limit=args.limit)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name="llava-qwen",
        device_map="auto",
    )
    set_kv_config(model)

    results = {"subPlot": [], "summary": []}
    for example in tqdm(dataset):
        pred = run_generation(
            model,
            tokenizer,
            image_processor,
            example,
            num_frames=args.num_frames,
            max_new_tokens=args.max_new_tokens,
        )
        row = {
            "video_name": example["video_name"],
            "Q": example["question"],
            "A": example["answer"],
            "pred": pred,
        }
        results[example["task_type"]].append(row)

    if "subPlot" in tasks:
        with open(output_dir / "subplot_all.json", "w") as f:
            json.dump(results["subPlot"], f, indent=2, ensure_ascii=False)
    if "summary" in tasks:
        with open(output_dir / "summary_all.json", "w") as f:
            json.dump(results["summary"], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
