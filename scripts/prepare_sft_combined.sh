#!/usr/bin/env bash
# prepare_sft_combined.sh — 3B SFT용 전체 데이터 통합
# 모든 SFT 데이터를 하나의 train/val 파일로 합침
#
# 업데이트 (2026-03-02): sft_extra 신규 소스 추가
#   - nayohan_Evol-Instruct-Code-80k-v1-ko  (코드 instruction)
#   - FreedomIntelligence_alpaca-gpt4-korean (GPT-4 alpaca 한국어)
#   - FreedomIntelligence_evol-instruct-korean (evol-instruct 한국어)
#   - coastral_korean-writing-style-instruct  (한국어 글쓰기 스타일)
#   - maywell_ko_wikidata_QA                  (위키데이터 QA)
#   - OpenAssistant_oasst1_ko                 (OASST1 한국어, 트리 재구성)
#   - Bllossom_evol-instruct-ko               (존재 확인 후 로드)
set -euo pipefail
BASE="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$BASE/data/sft_combined"
mkdir -p "$OUT_DIR"

python3 << 'PYEOF'
import json, random, os, glob, hashlib
from collections import defaultdict

BASE = "/PROJECT/0325120031_A/ghong/taketimes/llm-bang/data"
OUT_TRAIN = f"{BASE}/sft_combined/train.jsonl"
OUT_VAL = f"{BASE}/sft_combined/val.jsonl"
VAL_RATIO = 0.02
SEED = 42

# SFT 소스 파일 목록 (chat 포맷으로 변환 가능한 것들)
SOURCES = [
    # (path, fmt)  fmt: "messages" | "auto" | "oasst"
    (f"{BASE}/sft/train.jsonl", "messages"),
    (f"{BASE}/sft_extra/ultrachat_200k/train_sft.jsonl", "messages"),
    (f"{BASE}/sft_extra/open_korean_instructions/train.jsonl", "messages"),
    (f"{BASE}/sft_extra/korean_instruction_mix/train.jsonl", "messages"),
    (f"{BASE}/sft_extra/openhermes_2.5/train.jsonl", "messages"),
    (f"{BASE}/sft_extra/magpie_reasoning_v2/train.jsonl", "messages"),
    (f"{BASE}/sft_extra/magpie_reasoning_ko/train.jsonl", "messages"),
    (f"{BASE}/sft_extra/reasoning_r1_1.4m/train.jsonl", "messages"),
    (f"{BASE}/sft_extra/lemon-mint_smol-koreantalk.jsonl", "auto"),
    (f"{BASE}/sft_extra/dbdu_ShareGPT-74k-ko.jsonl", "auto"),
    (f"{BASE}/sft_extra/ko_lima/data.jsonl", "auto"),
    (f"{BASE}/sft_extra/koalpaca_v1_1a/data.jsonl", "auto"),
    (f"{BASE}/sft_extra/kullm_v2/data.jsonl", "auto"),
    (f"{BASE}/sft_extra/kuotient_orca-math-word-problems-193k-korean.jsonl", "auto"),
    (f"{BASE}/sft_extra/kyujinpy_KOR-OpenOrca-Platypus-v3/data.jsonl", "auto"),
    (f"{BASE}/sft_extra/nlp-with-deeplearning_Ko.WizardLM_evol_instruct_V2_196k.jsonl", "auto"),
    (f"{BASE}/sft_extra/AI-MO_NuminaMath-CoT/data.jsonl", "auto"),
    (f"{BASE}/sft_extra/zwhe99_DeepMath-103K/data.jsonl", "auto"),
    # ---- 신규 소스 (2026-03-02) ----
    (f"{BASE}/sft_extra/nayohan_Evol-Instruct-Code-80k-v1-ko/data.jsonl", "auto"),
    (f"{BASE}/sft_extra/FreedomIntelligence_alpaca-gpt4-korean.jsonl", "auto"),
    (f"{BASE}/sft_extra/FreedomIntelligence_evol-instruct-korean.jsonl", "auto"),
    (f"{BASE}/sft_extra/coastral_korean-writing-style-instruct.jsonl", "auto"),
    (f"{BASE}/sft_extra/maywell_ko_wikidata_QA.jsonl", "auto"),
    (f"{BASE}/sft_extra/OpenAssistant_oasst1_ko.jsonl", "oasst"),
    (f"{BASE}/sft_extra/Bllossom_evol-instruct-ko/data.jsonl", "auto"),
]

def to_messages(obj):
    """다양한 포맷을 통일된 messages 포맷으로 변환"""
    # 이미 messages 포맷
    if 'messages' in obj and isinstance(obj['messages'], list):
        return obj['messages']
    # conversations 포맷
    if 'conversations' in obj:
        msgs = []
        for turn in obj['conversations']:
            role = turn.get('from', turn.get('role', ''))
            content = turn.get('value', turn.get('content', ''))
            if role in ('human', 'user', 'prompter'):
                msgs.append({'role': 'user', 'content': content})
            elif role in ('gpt', 'assistant', 'bot'):
                msgs.append({'role': 'assistant', 'content': content})
        return msgs if len(msgs) >= 2 else None
    # instruction/output 포맷
    if 'instruction' in obj:
        instruction = obj['instruction']
        inp = obj.get('input', '')
        output = obj.get('output', obj.get('response', ''))
        if not output: return None
        user_content = instruction + ('\n\n' + inp if inp else '')
        return [{'role': 'user', 'content': user_content}, {'role': 'assistant', 'content': output}]
    # question/answer 포맷
    if 'question' in obj and 'answer' in obj:
        return [{'role': 'user', 'content': obj['question']}, {'role': 'assistant', 'content': obj['answer']}]
    # prompt/response
    if 'prompt' in obj and ('response' in obj or 'completion' in obj):
        resp = obj.get('response', obj.get('completion', ''))
        return [{'role': 'user', 'content': obj['prompt']}, {'role': 'assistant', 'content': resp}]
    # problem/solution
    if 'problem' in obj and 'solution' in obj:
        return [{'role': 'user', 'content': obj['problem']}, {'role': 'assistant', 'content': obj['solution']}]
    return None


def load_oasst(path):
    """
    OpenAssistant OASST1 flat message 포맷을 대화 트리로 재구성.
    각 루트(prompter) 메시지에서 best-ranked assistant 응답(rank=0.0)을
    따라 단일 대화 스레드를 추출한다.
    deleted=True 메시지와 review_result=False 메시지는 제외.
    """
    nodes = {}      # message_id → obj
    children = defaultdict(list)  # parent_id → [child_obj, ...]

    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get('deleted', False):
                continue
            if obj.get('review_result') is False:
                continue
            mid = obj.get('message_id')
            if mid:
                nodes[mid] = obj
            pid = obj.get('parent_id')
            if pid:
                children[pid].append(obj)

    # 자식 목록을 rank 오름차순 정렬 (rank=null은 뒤로)
    def sort_key(c):
        r = c.get('rank')
        mid = c.get('message_id', '')
        return (1, 0, mid) if r is None else (0, r, mid)
    for pid in children:
        children[pid].sort(key=sort_key)

    samples = []

    def build_thread(node, current_msgs):
        """재귀적으로 대화 스레드를 따라 samples에 추가."""
        role = node.get('role', '')
        text = node.get('text', '')
        if role == 'prompter':
            mapped_role = 'user'
        elif role == 'assistant':
            mapped_role = 'assistant'
        else:
            return

        msgs = current_msgs + [{'role': mapped_role, 'content': text}]

        # 유효한 user→assistant 쌍이 있을 때만 샘플 추가
        if mapped_role == 'assistant' and len(msgs) >= 2:
            samples.append({'messages': msgs})

        # 자식 중 best (rank=0.0) 하나만 따라간다 (가장 품질 높은 경로)
        kids = children.get(node.get('message_id'), [])
        if kids:
            build_thread(kids[0], msgs)

    # 루트 노드: parent_id가 없는 prompter 메시지
    roots = [n for n in nodes.values() if n.get('parent_id') is None and n.get('role') == 'prompter']
    for root in roots:
        build_thread(root, [])

    return samples


random.seed(SEED)
all_samples = []

for path, fmt in SOURCES:
    if not os.path.exists(path):
        print(f"[SKIP] {path}")
        continue

    if fmt == "oasst":
        samples = load_oasst(path)
        all_samples.extend(samples)
        print(f"[LOADED] {os.path.basename(path)}: {len(samples):,} samples (oasst tree)")
        continue

    count = 0
    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if fmt == "messages":
                msgs = obj.get('messages') or obj.get('conversations')
                if msgs:
                    all_samples.append({'messages': msgs})
                    count += 1
            else:  # auto detect
                msgs = to_messages(obj)
                if msgs and len(msgs) >= 2:
                    all_samples.append({'messages': msgs})
                    count += 1
    print(f"[LOADED] {os.path.basename(path)}: {count:,} samples")
    if count == 0:
        print(f"[WARN] {os.path.basename(path)}: 0 samples extracted (format detection may have failed)")

print(f"\n총 샘플: {len(all_samples):,}")

# ---- Deduplication (MD5 of first user message) ----
seen_hashes = set()
unique_samples = []
dup_count = 0
for s in all_samples:
    msgs = s.get('messages', [])
    first_user = next((m['content'] for m in msgs if m.get('role') == 'user'), '')
    h = hashlib.md5(first_user.encode('utf-8', errors='replace')).hexdigest()
    if h in seen_hashes:
        dup_count += 1
        continue
    seen_hashes.add(h)
    unique_samples.append(s)

print(f"[DEDUP] 제거: {dup_count:,}, 남은 샘플: {len(unique_samples):,}")
all_samples = unique_samples

# ---- Format validation ----
def validate_messages(msgs):
    """Check messages have valid role/content structure."""
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    for m in msgs:
        if not isinstance(m, dict):
            return False
        if m.get('role') not in ('user', 'assistant', 'system'):
            return False
        if not isinstance(m.get('content'), str):
            return False
    return True

valid_samples = []
invalid_count = 0
for s in all_samples:
    if validate_messages(s.get('messages', [])):
        valid_samples.append(s)
    else:
        invalid_count += 1

print(f"[VALIDATE] 유효하지 않은 포맷 제거: {invalid_count:,}, 남은 샘플: {len(valid_samples):,}")
all_samples = valid_samples

random.shuffle(all_samples)

n_val = int(len(all_samples) * VAL_RATIO)
val_samples = all_samples[:n_val]
train_samples = all_samples[n_val:]

os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)
with open(OUT_TRAIN, 'w') as f:
    for s in train_samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
with open(OUT_VAL, 'w') as f:
    for s in val_samples:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')

print(f"[DONE] train: {len(train_samples):,} → {OUT_TRAIN}")
print(f"[DONE] val:   {len(val_samples):,} → {OUT_VAL}")
PYEOF
echo "SFT 데이터 병합 완료"
