import sys
import os
sys.path.append(os.path.abspath("../LongVA")) # LongVA 경로

import torch
import json
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu

from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token
from longva.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from longva.conversation import conv_templates

# LongVA 기본 비디오 로딩 함수 (MLVU 논문 기준 256 프레임 추출)
def load_video(video_path, num_frames=256):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    video = vr.get_batch(frame_indices).asnumpy()
    return video


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret


class MLVU(Dataset):
    def __init__(self, json_path, video_base_dir):
        self.data_list = []
        # test_mcq_gt.json (정답지) 하나만 읽어옵니다.
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for data in json_data:
            self.data_list.append({
                'task_type': data['question_type'], 
                'video_path': os.path.join(video_base_dir, data['video']),
                'data': data
            })
            
    def __str__(self):
        len_list = {}
        option_list = {}
        for item in self.data_list:
            task = item['task_type']
            if task not in len_list:
                len_list[task] = 0
                option_list[task] = 0
            len_list[task] += 1
            option_list[task] += len(item['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            # 타입 에러 방지 (숫자형 정답 대비)
            if str(c) == str(answer):
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = self.data_list[idx]['video_path']
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_path, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }



def check_ans(pred, gt):
    flag = False

    index=gt.index("(")
    index2=gt.index(")")
    gt_option=gt[index+1:index2]

    if ")" in pred:
        index3=pred.index(")")
        pred=pred[index3-1:index3]

    if pred==gt_option:
        flag=True

    return flag

def main():

    # Test 셋 비디오가 모여있는 폴더 경로
    video_base_dir = "/NHNHOME/WORKSPACE/0226010268_A/yhlee/MLVU/data/MLVU_videos/MLVU_Test/video"
    
    # 정답지가 포함된 Ground Truth JSON 경로
    json_path = "/NHNHOME/WORKSPACE/0226010268_A/yhlee/MLVU/data/MLVU_videos/test-ground-truth/test_mcq_gt.json"
    
    save_path = f"./test_all_choice"
    result_path = f"bench_all.json"

    dataset = MLVU(json_path, video_base_dir)


    '''
    load your model
    '''
    from longva.model.builder import load_pretrained_model
    
    # V-NIAH와 달리 MLVU는 비디오 원본을 처리해야 하므로 반드시 builder.py를 사용해야 합니다.
    # 불확실한 로컬 경로 대신 허깅페이스 공식 리포지토리에서 확실하게 가중치를 불러옵니다.
    model_path = "lmms-lab/LongVA-7B" 
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-qwen", # builder.py 내부의 'llava' 문자열 검사를 통과하기 위한 트릭 유지
        device_map="auto"
    )

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    
    for example in tqdm(dataset):
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0]
        acc_dict[task_type][1] += 1
        total += 1
        
        video_path = example["video"]
        quesiotn = example["question"]

        video_np = load_video(video_path, num_frames=256)
        video_tensor = image_processor.preprocess(video_np, return_tensors='pt')['pixel_values'].half().to(model.device)
        if isinstance(video_tensor, torch.Tensor):
            video_tensor = [video_tensor]

        conv_mode = "qwen_1_5"
        conv = conv_templates[conv_mode].copy()
        conv.system = "Carefully watch this video and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question."
        
        inp = DEFAULT_IMAGE_TOKEN + '\n' + quesiotn + " Only give the best option."
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], "Best Option: (")
        prompt = get_prompt2(conv)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        model.config.kv_mode = 3
        model.config.kv_budget = 1024
        model.config.kv_window = 32
        model.config.kv_sink = 4
        model.config.kv_alpha = 0.5
        model.config.kv_tau = 3.0
        model.config.kv_gamma = 0.9
        
         
        # 5. 추론 실행 (🔥 V-NIAH 완벽 이식: 2048 청크 우회 및 한 번에 밀어넣기)
        with torch.inference_mode():
            # (1) 입력 전처리
            (new_input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = \
                model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, video_tensor, ["video"], None
                )
            
            # 마지막 질문 예측 토큰 1개만 분리
            split_idx = inputs_embeds.shape[1] - 1
            context_embeds = inputs_embeds[:, :split_idx, :]
            trigger_embeds = inputs_embeds[:, split_idx:, :]

            # (2) 문맥 1-Pass 통과 및 압축 (Question-Aware Eviction 완벽 발동)
            model.config.chunked_prefill_evict = True
            out_context = model(inputs_embeds=context_embeds, use_cache=True, return_dict=True)
            model.config.chunked_prefill_evict = False
            
            past_key_values = out_context.past_key_values
            kv_len = past_key_values[0][0].shape[-2] if isinstance(past_key_values, (list, tuple)) else past_key_values.get_seq_length()
            
            # (3) 오프셋 설정 및 첫 번째 예측 트리거
            model.config.pos_offset = split_idx - kv_len
            
            trigger_mask = torch.ones((1, kv_len + 1), device=model.device, dtype=torch.long)
            trigger_pos = torch.tensor([[split_idx]], device=model.device, dtype=torch.long)
            
            out_trigger = model(
                inputs_embeds=trigger_embeds,
                attention_mask=trigger_mask,
                position_ids=trigger_pos,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            # (4) 압축된 캐시를 바탕으로 나머지 텍스트 생성 (Decoding Jitter 방지)
            next_token_id = out_trigger.logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            past_key_values = out_trigger.past_key_values
            kv_len_trigger = past_key_values[0][0].shape[-2] if isinstance(past_key_values, (list, tuple)) else past_key_values.get_seq_length()
            
            gen_mask = torch.ones((1, kv_len_trigger + 1), device=model.device, dtype=torch.long)
            
            original_budget = getattr(model.config, "kv_budget", 999999)
            model.config.kv_budget = 9999999 # 디코딩 중 추가 Evict 차단
            
            # llava_qwen.py 내부의 NoneType 에러를 방지하기 위해 
            # 인자 이름을 'inputs'와 'input_ids' 모두 명시적으로 전달합니다.
            output_ids = model.generate(
                inputs=next_token_id,
                input_ids=next_token_id,
                attention_mask=gen_mask,
                past_key_values=past_key_values,
                max_new_tokens=10,
                use_cache=True,
                do_sample=False,
                temperature=0.0
            )
            
            # 원상 복구
            model.config.pos_offset = 0
            model.config.kv_budget = original_budget
            
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        
        
        
    
        gt = example['answer']
        res_list.append({
            'pred': pred,
            'gt': gt,
            'question':example['question'],
            'question_type':example['task_type'],
            'video':example['video']
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)


    with open(f"{save_path}.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    total=0
    idx=0
    for k, v in acc_dict.items():
        idx+=1
        final_res[k] = v[0] / v[1] * 100  
        total+=final_res[k]
    final_res['Avg'] = total /idx 
    print(final_res)

    with open(result_path, "w") as f:
        json.dump(final_res, f)


if __name__ == '__main__':
    main()
