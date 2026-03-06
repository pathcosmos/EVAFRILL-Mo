# 한국어 LLM 전체 로드맵 & 의사결정 프레임워크

> **작성일**: 2026-02-26  
> **현재 상태**: SFT 5,000 steps 완료 (loss 1.9677), 1.19B 파라미터 모델  
> **목표**: 실사용 가능한 한국어 LLM 배포

---

## 0. TL;DR — 지금 당장 할 일

1. **SFT 모델 빠른 생성 테스트** (30분, 오늘): temperature sampling으로 반복 퇴화 확인
2. **lm-eval-harness ko_ifeval + ko_winogrande 실행** (2~4시간): 숫자 확인
3. **결과에 따라 분기** → 아래 의사결정 트리 참조

---

## 1. 현재 위치 파악

### 1.1 SFT 학습 현황 요약

| 항목 | 값 |
|------|-----|
| Steps | 5,000 |
| Final Loss | 1.9677 |
| 학습 시간 | 0.61h (~37분) |
| 처리 속도 | ~75,700 tok/s (단일 B200) |
| LR (final) | 2.00e-06 (완전히 decay됨) |
| Gradient Norm | 안정 (1.0~1.4 범위) |
| SFT 데이터 | 알 수 없음 (확인 필요) |

**주목**: 5,000 steps는 **매우 적은 양**이다. SFT에서 보통 1~3 에폭을 돌리는데, 데이터셋 크기에 따라 steps 충분성이 결정된다.

### 1.2 업계 내 위치

#### 벤치마크 기준 (Open Ko-LLM Leaderboard 실측치 기반 추정)

```
모델                          규모      ko_ifeval   ko_winogrande   비고
──────────────────────────────────────────────────────────────────────
EXAONE-3.0-7.8B-Instruct     7.8B      ~55%        ~80%+           8T tokens, SFT+DPO
Llama-3.1-8B-Korean-SFT      8B        ~40%        ~72%            Llama 기반 한국어 적응
SOLAR-10.7B-Instruct          10.7B     ~50%        ~78%            업스테이지
Gemma-2-9B-Korean             9B        ~45%        ~75%            Google 기반

── 현실적 1B SFT 벤치마크 (타사 사례) ────────────────────────────────
dltjdgh0928/test_instruction  ~?B       24.1%       57.1%           리더보드 실측
lookuss/test-llilu             ~?B       22.9%       58.2%           리더보드 실측
generic 1B SFT (추정)          1~2B      20-30%      52-62%          현실적 범위

── 우리 모델 예상 ──────────────────────────────────────────────────────
korean_1b_sft (5k steps)      1.19B     15-28%?     50-58%?         평가 전 추정
```

#### 핵심 격차 분석

| 비교 대상 | 파라미터 차이 | 예상 성능 격차 | 주요 이유 |
|-----------|--------------|---------------|-----------|
| EXAONE-3.0-7.8B | 6.6× | 매우 큼 | 규모 + 데이터 + DPO |
| 8B 한국어 SFT | 6.7× | 큼 | 규모 차이가 지배적 |
| 타사 1B SFT | 유사 | 작음~중간 | 데이터/학습 방법 차이 |

**현실적 평가**: 1B 모델은 7~10B 모델과 **direct 경쟁이 불가능**하다. 그러나:
- **엣지 배포** (로컬 서빙, 저지연 API): 1B가 명확한 우위
- **리소스 효율**: 1B는 단일 GPU, 심지어 CPU에서도 구동
- **특화 도메인**: 한국어 특화 fine-tuning으로 특정 태스크에서 대형 범용 모델 근접 가능

### 1.3 1B 규모의 한계와 가능성

**현실적 한계**:
- ko_ifeval 30% 초과 어려움 (instruction following 복잡도)
- 수학/코드: 사전학습 데이터 없으면 거의 불가
- 장문 맥락 이해: 4K context에서 degradation 시작
- 사실 기억: 세밀한 사실 저장 capacity 부족

**가능한 것**:
- 한국어 기본 QA, 요약, 분류
- 간단한 지시 따르기 (1~2단계)
- 한국어 자동완성, 교정
- 도메인 특화 태스크 (제한된 형식)

---

## 2. 단계별 로드맵

### Phase 1: SFT 검증 (지금 → ~1주)

#### 목표
SFT 5,000 steps 결과가 실사용 가능한 수준인지 판정

#### 체크리스트

```
□ 1-1. 생성 품질 빠른 점검 (30분)
   - temperature=0.8, top_p=0.9으로 10개 프롬프트 생성
   - 체크: 반복 퇴화 비율 (목표: < 20%)
   - 체크: 한국어 어미/조사 처리 자연스러운가
   - 체크: instruction 따르는가 (base와 비교)

□ 1-2. 공식 벤치마크 (2~4시간)
   - lm-evaluation-harness 설치 및 ko_ifeval 실행
   - lm-evaluation-harness ko_winogrande 실행
   - 선택: ko_gsm8k (수학 데이터 없으면 skip 가능)

□ 1-3. SFT 데이터 품질 점검
   - SFT 학습에 사용된 데이터셋 확인
   - 데이터 수 (몇 개 샘플인가?)
   - 5,000 steps × batch_size = 총 토큰 수 산출
   - 에폭 수 계산: epoch 2에 진입했으므로 최소 1 에폭 완료 확인됨

□ 1-4. Base vs SFT 비교
   - 동일 프롬프트에 base (pretrained)와 SFT 결과 비교
   - SFT가 instruction following 능력을 부여했는가?
```

#### Pass/Fail 기준 (수치화)

| 지표 | Pass ✅ | 경계선 ⚠️ | Fail ❌ |
|------|---------|-----------|---------|
| ko_ifeval (prompt strict) | > 25% | 15~25% | < 15% |
| ko_winogrande | > 53% | 50~53% | < 50% |
| 반복 퇴화율 (greedy) | < 20% | 20~40% | > 40% |
| temperature 샘플링 품질 | 자연스러움 | 어색함 | 무의미 |
| Base 대비 SFT 개선 | 명확한 instruction 따르기 | 미미한 개선 | 개선 없음/악화 |

> **참고**: ko_winogrande 50%는 random (binary choice) 수준. 실질적 의미 있으려면 53%+.

#### 실패 시 대응

- **ko_ifeval < 15% + 반복 > 40%**: SFT 데이터 문제 또는 steps 부족 → Phase 2A
- **Base 대비 개선 없음**: SFT 데이터 형식/품질 점검, 학습률 재검토
- **모든 지표 Fail**: 데이터 파이프라인부터 재검토

---

### Phase 2A: SFT 개선 (선택적, Phase 1 결과 기준)

#### 언제 진입하는가?

```
Phase 2A 진입 조건:
├── ko_ifeval < 25% AND 반복 > 20%     → 즉시 진입
├── ko_ifeval 25-30% AND 반복 20-30%   → 데이터 보강 후 진입
└── ko_ifeval > 30% AND 반복 < 15%     → Phase 2B 또는 4로 바로
```

#### 옵션별 분석

**옵션 A: Steps 증가 (5k → 10k~20k)**
- **언제**: 데이터는 충분하고 아직 수렴하지 않은 경우
- **확인 방법**: Loss 곡선이 아직 하강 중인가? (5,000 steps에서 1.97 — 수렴 근접)
- **예상 효과**: 소폭 개선 (loss 1.97 → 1.80 목표, ko_ifeval +3~7%p 예상)
- **비용**: B200 1개 기준 1.5~3시간 추가
- **주의**: 이미 epoch 2에 진입 — 과적합 위험 있음

**옵션 B: 더 좋은 데이터**
- **언제**: 현재 SFT 데이터가 부족하거나 품질이 낮을 때 (가장 흔한 이유)
- **추천 데이터셋**:
  - `beomi/KoAlpaca-v1.1a` — 21K 한국어 instruction
  - `HAERAE-HUB/KMMLU` — 한국어 지식 QA
  - `nayohan/llama3-baseline-ko-dataset` — 다양한 instruction
  - `squarelike/sharegpt_deepl_ko_ko-en` — ShareGPT 한국어
  - 합산 목표: 50K~200K 고품질 샘플
- **예상 효과**: 데이터 품질 개선이 steps 2배보다 효과적 (경험적 법칙)
- **비용**: 데이터 정제 1~2일, 학습 추가 2~6시간

**옵션 C: ORPO (Odds Ratio Preference Optimization)**
- **언제**: SFT baseline 확보 후 preference 정렬이 필요할 때
- **장점**: reference model 불필요 → 메모리 절약, 학습 단순화
- **한국어 데이터**: `kuotient/orca-math-korean-preference` (193K), `heegyu/orca-math-korean-preference-cleaned` (192K) 존재
- **예상 효과**: 반복 퇴화 -10~20%p, instruction following +5~10%p
- **비용**: 데이터 준비 1일, 학습 3~6시간

**옵션 D: DPO (Direct Preference Optimization)**
- **언제**: ORPO보다 더 강한 정렬이 필요할 때, 또는 SFT가 어느 정도 잘 됐을 때
- **장점**: RLHF와 유사한 효과, PPO보다 안정적
- **단점**: reference model 필요 (메모리 2×)
- **B200에서 가능성**: 1.19B × 2 = ~2.4B params — 단일 B200 183GB에서 충분
- **비용**: 학습 4~8시간

#### 권장 순서
```
데이터 점검 → 데이터 보강 (옵션 B) → Steps 추가 (옵션 A) → ORPO (옵션 C)
```

---

### Phase 2B: 스케일업 — 3B 모델

#### 데이터 충분성 분석

| 기준 | 필요 토큰 | 현재 보유 | 판정 |
|------|-----------|-----------|------|
| Chinchilla 최소 (20×) | 3B × 20 = 60B | ~150B | ✅ 충분 |
| Chinchilla 최적 (70×) | 3B × 70 = 210B | ~150B | ⚠️ 71% 수준 |
| Llama 방식 (고품질 집중) | 3B × 100 = 300B | ~150B | ❌ 부족 |

**결론**: **지금 데이터로 3B 학습 가능**. 단, optimal은 아님. 고품질 데이터를 50B 추가 수집하면 optimal 근접.

#### 예상 학습 시간 (8× B200 기준)

```
3B 모델 설정 추정:
- 처리 속도: ~2.5~3M tok/s (8× B200, 1.19B 기준 2.64M)
  → 3B 모델은 속도 ~40% 감소 예상 (메모리/연산 증가)
  → 실효 속도: ~1.6M tok/s (추정)

60B tokens (최소): 60B / 1.6M = 37,500초 ≈ 10.4시간
150B tokens (현재 보유 전량): 150B / 1.6M = 93,750초 ≈ 26시간
210B tokens (optimal): 210B / 1.6M = 131,250초 ≈ 36.5시간

→ 현실적 학습 기간: 1~2일 (8× B200)
```

#### 3B 학습 준비사항

```
□ 모델 아키텍처 설정 변경:
  - d_model: 2048 → 2560 (또는 3072)
  - n_layers: 24 → 32
  - n_heads: 16 → 32
  - n_kv_heads (GQA): 4 → 8
  - d_ffn: 5472 → ~8192
  → 예상 파라미터: ~3B

□ 데이터 준비:
  - cc100 ko 재다운로드 (버그 수정 후)
  - CulturaX 24.8B 활용
  - 총 150B+ 토큰 한국어 데이터 병합

□ configs/korean_3b_fp8.yaml 작성
□ 체크포인트 저장 전략: 매 5,000 steps
□ FP8 설정 유지 (B200 최적화)
```

#### 1B SFT 결과의 3B 진행 여부 영향

```
1B SFT 결과              → 3B 진행 여부
──────────────────────────────────────────────────────
ko_ifeval > 30%          → 강력히 추천: 1B가 이미 좋음, 3B는 확실히 더 좋을 것
ko_ifeval 20-30%         → 조건부 추천: 데이터/방법론 확인 후 3B
ko_ifeval < 20%          → 3B 전에 원인 분석 필수: 같은 문제가 3B에도 재현됨
반복 퇴화 > 40%          → 사전학습 데이터 문제 의심: 3B도 동일 문제 가능
SFT 개선 없음            → SFT 파이프라인 수정 후 3B
```

---

### Phase 3: RLHF / Preference Optimization (선택적)

#### 언제 필요한가?

| 시나리오 | 필요성 |
|----------|--------|
| 서비스 배포 (사용자 대면) | 강력히 필요 — safety, coherence |
| 리더보드 점수 극대화 | 필요 — DPO/ORPO로 +5~15%p |
| 내부 연구/실험 | 불필요 |
| RAG 시스템 백엔드 | 불필요 |

#### ORPO vs DPO vs PPO 비교

| 방법 | 언제 | 메모리 | 복잡도 | 한국어 데이터 |
|------|------|--------|--------|---------------|
| **ORPO** | SFT와 동시, 빠른 정렬 | 1× (ref 없음) | 낮음 | 193K+ 존재 |
| **DPO** | SFT 이후, 안정적 정렬 | 2× (ref 필요) | 중간 | 193K+ 존재 |
| **SimPO** | ref 없이 DPO 효과 | 1× | 중간 | 범용 적용 |
| **PPO** | RLHF 완전 구현 | 3~4× | 높음 | reward model 필요 |

**B200 환경에서 추천**: ORPO 또는 SimPO (reference model 없음, 메모리 효율)

#### 한국어 Preference 데이터 현황 (HuggingFace)

```
kuotient/orca-math-korean-preference     193K 샘플  수학 중심
heegyu/orca-math-korean-preference-cleaned  192K   수학 (정제본)
lemon-mint/korean-realqa-reasoning-v01-preference  7.7K  추론
ChuGyouk/argilla-distilabel-math-preference-dpo-korean  2.4K  소규모

→ 수학 특화 데이터가 많음. 일반 한국어 preference는 부족.
→ 일반 preference는 자체 생성 또는 번역으로 보강 필요.
  방법: GPT-4/Claude로 chosen/rejected 쌍 생성 (Self-Play)
```

---

### Phase 4: 배포

#### 서빙 옵션 비교

| 옵션 | 특징 | B200 적합성 | 추천 상황 |
|------|------|-------------|-----------|
| **vLLM** | PagedAttention, 고처리량 | ✅ 최우수 | API 서버, 배치 추론 |
| **TGI (Text Generation Inference)** | HF 공식, 안정적 | ✅ 우수 | HF Hub 연동 |
| **llama.cpp + GGUF** | CPU/저사양 가능 | ⚠️ B200에선 과소 | 엣지 배포, Ollama |
| **Ollama** | 로컬 배포 편의성 | ⚠️ | 개인 사용, 데모 |

**B200 기준 vLLM 예상 throughput (1.19B 모델)**:

```
1.19B 모델 (BF16):
  - 메모리: ~2.4GB (파라미터) + KV cache
  - 단일 B200 183GB: KV cache 극대화 가능
  - 예상 throughput: 5,000~15,000 tokens/s (배치 처리)
  - 단일 스트리밍: 200~500 tokens/s (사용자 체감)
  → 동시 사용자 100~500명 지원 가능 (단일 GPU)
```

#### 양자화 옵션 (B200 환경)

| 포맷 | 정밀도 손실 | 크기 | B200 적합성 | 추천 |
|------|------------|------|-------------|------|
| FP8 (Native) | 없음 | 1.2GB | ✅ 최우수 (HW 지원) | **최우선** |
| BF16 | 없음 | 2.4GB | ✅ 기본 | 기준선 |
| AWQ (W4A16) | 매우 적음 | 0.6GB | ✅ 우수 | 엣지/저메모리 |
| GPTQ (W4) | 적음 | 0.6GB | ✅ 우수 | CPU 오프로드 |
| GGUF Q4_K_M | 적음 | ~0.7GB | ⚠️ (CPU용) | Ollama 배포용 |

**B200 권장**: FP8 → AWQ 순서로 고려. B200은 FP8 하드웨어 지원으로 양자화 없이 이미 효율적.

#### HuggingFace Hub 업로드

```
필요 작업:
□ HF 포맷 변환: config.json, model.safetensors, tokenizer_config.json
□ model card 작성 (한국어 설명, 벤치마크 결과, 사용법)
□ 라이선스 설정 (Apache 2.0 권장)
□ eval 결과 포함
□ Open Ko-LLM Leaderboard 제출 (평가 요청)
```

---

## 3. 의사결정 트리 (수치 기반)

```
══════════════════════════════════════════════════════════════════
                    [Phase 1: SFT 평가 결과]
══════════════════════════════════════════════════════════════════

├── ko_ifeval > 30% AND 반복율 < 15%
│   ├── 데이터 150B 모두 사용 가능? → Phase 2B (3B 사전학습)
│   └── 지금 당장 배포가 목표? → Phase 4 (vLLM 서빙 + HF 업로드)
│
├── ko_ifeval 20~30% AND 반복율 15~30%
│   ├── SFT 데이터가 < 10K 샘플? → Phase 2A-B (데이터 보강 최우선)
│   ├── SFT 데이터가 10~50K 샘플? → Phase 2A-A (steps 추가) + 2A-C (ORPO)
│   └── SFT 데이터가 > 50K 샘플? → Phase 2A-A (steps 추가) OR 2B (3B)
│
├── ko_ifeval 10~20% AND 반복율 30~50%
│   ├── base 모델과 SFT 차이 없음? → SFT 파이프라인 버그 점검
│   ├── SFT 데이터 품질 의심? → 데이터 전수 점검 후 Phase 2A-B
│   └── base PPL이 높음 (> 15)? → 사전학습 더 필요 (데이터 추가)
│
└── ko_ifeval < 10% OR 반복율 > 50%
    ├── base 모델 자체가 이미 반복 > 30%? → 사전학습 데이터 품질 문제
    │   └── → cc100 노이즈 필터링 후 추가 사전학습
    ├── SFT loss가 발산했는가? → 학습률/optimizer 설정 재검토
    └── 모든 생성이 무의미? → 체크포인트 손상 확인, 이전 체크포인트 복원

══════════════════════════════════════════════════════════════════
                 [Phase 2A 내부 의사결정]
══════════════════════════════════════════════════════════════════

Phase 2A 진입 후:

├── 현재 SFT 데이터 < 20K 샘플?
│   └── → 데이터 보강이 steps 추가보다 효과적 (최우선)
│       데이터: beomi/KoAlpaca, squarelike/sharegpt_ko, nayohan/llama3-ko
│
├── loss curve가 아직 하강 중 (step 4000~5000 차이 > 0.05)?
│   └── → steps 2배 추가 시도 (10k까지)
│
├── 반복율 > 30% (주요 문제)?
│   └── → ORPO 또는 repetition penalty 적용 먼저
│       ORPO 데이터: kuotient/orca-math-korean-preference (193K)
│
└── ko_ifeval < 20% + 데이터 보강 후에도 개선 없음?
    └── → 3B 사전학습으로 전환 (1B SFT 한계 도달 가능성)

══════════════════════════════════════════════════════════════════
                 [Phase 2B 내부 의사결정]
══════════════════════════════════════════════════════════════════

3B 사전학습 진행 결정 시:

├── 현재 150B 토큰이 한국어 단일 언어?
│   └── → 영어 데이터 10~30% 혼합 권장 (cross-lingual transfer)
│       영어 수학/코드 포함하면 ko_gsm8k 등 추가 개선 가능
│
├── cc100 ko 데이터 수집 완료?
│   └── No → CulturaX 24.8B만으로 시작 가능 (60B 목표 달성 가능)
│
└── 3B 학습 중 중간 checkpoint에서 SFT 테스트?
    └── → 1B보다 3B base가 SFT 반응성이 높으면 3B SFT로 바로 진행

══════════════════════════════════════════════════════════════════
                    [Phase 4 배포 의사결정]
══════════════════════════════════════════════════════════════════

배포 방식 선택:

├── 연구/데모 목적?
│   └── → HF Hub 업로드 + Gradio Space 생성 (무료)
│
├── 내부 API 서빙?
│   └── → vLLM (FP8 native) + OpenAI 호환 엔드포인트
│       커맨드: vllm serve ./checkpoints/korean_1b_sft --dtype fp8
│
├── 개인/팀 로컬 사용?
│   └── → GGUF Q4_K_M 변환 + Ollama (이미 Modelfile 존재)
│
└── Open Ko-LLM 리더보드 등재?
    └── → HF Hub 업로드 필수 → 리더보드 제출 양식 작성
```

---

## 4. 추가 확장 Job 후보군 (우선순위 순)

### 즉시 가능 (지금 서버에서 바로, 추가 데이터 불필요)

| 우선순위 | Job | 예상 시간 | 기대 효과 |
|----------|-----|-----------|-----------|
| ⭐⭐⭐ | **SFT 모델 생성 테스트** (temperature sampling) | 30분 | 반복율 현황 파악 |
| ⭐⭐⭐ | **lm-eval-harness 설치 + ko_ifeval 실행** | 2~4시간 | 공식 벤치마크 수치 |
| ⭐⭐⭐ | **ko_winogrande 실행** | 1~2시간 | 언어 이해 수치 |
| ⭐⭐ | **Base vs SFT 비교 생성** (동일 프롬프트) | 1시간 | SFT 효과 측정 |
| ⭐⭐ | **SFT 학습 손실 곡선 분석** (tensorboard) | 30분 | 수렴 여부 판단 |
| ⭐⭐ | **반복 퇴화 정량 측정** (repetition_penalty 효과) | 1시간 | 배포 가능성 판단 |
| ⭐ | **vLLM 서빙 테스트** (FP8) | 1~2시간 | throughput 측정 |
| ⭐ | **HF 포맷 변환** (config.json, safetensors) | 2~3시간 | HF Hub 업로드 준비 |

### 데이터 준비 필요

| 우선순위 | Job | 준비 시간 | 기대 효과 |
|----------|-----|-----------|-----------|
| ⭐⭐⭐ | **SFT 데이터 보강** (KoAlpaca + ShareGPT-ko 50K~) | 1~2일 | ko_ifeval +5~15%p |
| ⭐⭐⭐ | **cc100 재수집** (버그 수정 후) | 0.5~1일 | 150B+ 토큰 확보 |
| ⭐⭐ | **ORPO 데이터 준비** (orca-math-korean 193K) | 0.5일 | 반복 퇴화 -20%p |
| ⭐⭐ | **3B 사전학습 데이터 병합** (150B 토큰 통합) | 1~2일 | 3B 학습 준비 |
| ⭐ | **일반 한국어 preference 데이터 생성** (GPT-4 활용) | 3~7일 | 범용 ORPO/DPO |
| ⭐ | **영어/코드 데이터 추가** (10~30% 혼합) | 1~3일 | 수학/코드 개선 |

### 외부 리소스 필요

| 우선순위 | Job | 필요 리소스 | 기대 효과 |
|----------|-----|-------------|-----------|
| ⭐⭐ | **HuggingFace Hub 계정 업로드** | HF 계정, 인터넷 | 리더보드 제출 가능 |
| ⭐⭐ | **Open Ko-LLM Leaderboard 제출** | HF 계정 | 공식 순위 확인 |
| ⭐ | **KoMT-Bench / LogicKor 평가** | 외부 API 또는 스크립트 | 질적 평가 |
| ⭐ | **VRAM 증설 또는 Multi-GPU SFT** | 현재 12GB → 가능 더 필요? | 더 큰 배치 |

---

## 5. 리스크 분석

### 5.1 현재 학습 방식의 잠재적 문제점

| 리스크 | 심각도 | 현재 증거 | 완화 방법 |
|--------|--------|-----------|-----------|
| SFT steps 과소 (5k) | 🔴 높음 | epoch 2 진입, loss 아직 1.97 | steps 증가 또는 데이터 보강 |
| 사전학습 데이터 부족 (~8.91B) | 🟡 중간 | Chinchilla 대비 1B × 20 = 20B 필요 → 미달 | 150B 데이터 추가 학습 |
| 코드/수학 데이터 없음 | 🟡 중간 | ko_gsm8k 거의 0 예상 | 영어 코드/수학 데이터 혼합 |
| Greedy decoding 반복 퇴화 | 🔴 높음 | base에서 30% 발생 확인 | SFT + repetition_penalty + ORPO |

### 5.2 cc100 데이터 품질 이슈

**알려진 문제**:
- cc100은 CommonCrawl에서 추출된 웹 텍스트로 **노이즈가 심함**
- 한국어 cc100 특히: 광고 텍스트, 스팸, 반복 콘텐츠 다수
- 중복률: 문서 수준 중복 10~30% 추정 (MinHash 제거 필요)

**실제 영향**:
```
노이즈 포함 학습 → 모델이 광고/스팸 패턴 학습 → 생성 품질 저하
중복 데이터 → 특정 패턴 과도 암기 → 반복 퇴화 악화
```

**권장 전처리**:
```bash
# 1. 중복 제거 (MinHash LSH)
python scripts/dedup_minhash.py --input cc100_ko.bin --threshold 0.8

# 2. 품질 필터링 (perplexity 기반)
# 낮은 품질 텍스트: PPL > 1000 제거
python scripts/quality_filter.py --max_ppl 1000

# 3. 길이 필터링
# 너무 짧은 문장 (< 50 tokens) 제거
```

### 5.3 Tokenizer 선택 (korean_sp 64K)의 영향

**현재 설정**: SentencePiece Unigram 64K vocab, 한국어 특화

**장점**:
- 한국어 형태소 분리에 최적화 → 효율적 인코딩
- 64K vocab으로 영어 vs 한국어 token fertility 균형
- 한국어 글자 1개 = 평균 1.2~1.8 tokens (BPE 대비 효율적)

**잠재적 문제**:
| 문제 | 심각도 | 설명 |
|------|--------|------|
| 영어 vocabulary 부족 | 🟡 중간 | 영어 코드/수학 처리 효율 낮음 (byte fallback) |
| 기존 모델과 호환 불가 | 🟡 중간 | RLHF 데이터 재토크나이징 필요 |
| 신조어/외래어 처리 | 🟡 중간 | OOV 처리는 byte fallback이지만 느림 |
| 표준 Llama/Mistral 토크나이저와 다름 | 🟢 낮음 | HF 업로드 시 tokenizer 포함하면 OK |

**완화**:
- 향후 3B 모델에서는 **tiktoken (cl100k_base) 또는 Llama 계열 토크나이저 채택** 고려
- 현재 1.19B 모델은 현재 토크나이저 유지 (재학습 비용 too high)

---

## 6. 시나리오 목록 ("만약 X라면 Y를 해야 한다")

| # | 조건 (IF) | 액션 (THEN) |
|---|-----------|-------------|
| 1 | ko_ifeval > 30% AND 반복 < 15% | → 즉시 HF Hub 업로드 + 리더보드 제출 + 3B 사전학습 병렬 진행 |
| 2 | ko_ifeval 20~30% AND 반복 15~30% | → KoAlpaca+ShareGPT-ko로 데이터 보강 후 10k steps SFT 재실행 |
| 3 | ko_ifeval < 20% AND base와 차이 없음 | → SFT 학습 파이프라인 버그 점검 (데이터 로딩, 포맷 확인) |
| 4 | 반복율 > 40% | → ORPO (orca-math-korean 193K) 즉시 적용 |
| 5 | 모든 SFT 시도 후에도 ko_ifeval < 20% | → 1B 한계 인정, 3B 사전학습으로 전환 |
| 6 | cc100 수집 완료 (65~100B) | → 3B 사전학습 바로 시작 (26시간, 8× B200) |
| 7 | 3B base PPL < 8 달성 | → 3B SFT (KoAlpaca + ORPO) → 리더보드 목표 ko_ifeval 40%+ |
| 8 | 서비스 배포 결정 | → vLLM FP8 서빙 + GGUF Q4_K_M Ollama 병행 |
| 9 | 수학/코드 성능 필요 | → 영어 수학+코드 데이터 20% 혼합하여 3B 재학습 |
| 10 | 한국어 preference 데이터 자체 생성 원함 | → Claude/GPT-4로 chosen/rejected 쌍 10K 생성 후 DPO |

---

## 7. 전체 타임라인

```
현재 (2026-02-26)
│
├─ Week 1: Phase 1 검증
│  ├─ D+0: SFT 생성 테스트 (30분)
│  ├─ D+0: lm-eval ko_ifeval + ko_winogrande (4시간)
│  └─ D+2: 결과 분석 + 다음 단계 결정
│
├─ Week 2~3: Phase 2A 또는 2B 결정 후 실행
│  ├─ [2A 경로] 데이터 보강 (3~5일) + 재학습 (1~2일)
│  └─ [2B 경로] 3B 사전학습 (26시간) + 3B SFT (3~6시간)
│
├─ Week 4: Phase 3 (필요시)
│  └─ ORPO 학습 (193K 데이터, 3~6시간)
│
└─ Week 4~5: Phase 4 배포
   ├─ HF 포맷 변환 (2~3시간)
   ├─ HF Hub 업로드 + Model Card
   ├─ vLLM 서빙 설정
   └─ Ko-LLM 리더보드 제출

총 예상 기간: 3~5주 (3B 스케일업 포함)
```

---

## 8. 즉각적인 다음 단계 (Action Items)

```bash
# Step 1: lm-evaluation-harness 설치
pip install lm-eval

# Step 2: ko_ifeval 실행 (SFT 체크포인트)
lm_eval \
  --model hf \
  --model_args pretrained=/PROJECT/0325120031_A/ghong/taketimes/llm-bang/checkpoints/korean_1b_sft/checkpoint-0005000,dtype=bfloat16 \
  --tasks ko_ifeval \
  --device cuda:0 \
  --output_path ./eval/results/sft_5k_ko_ifeval.json

# Step 3: ko_winogrande 실행
lm_eval \
  --model hf \
  --model_args pretrained=/PROJECT/0325120031_A/ghong/taketimes/llm-bang/checkpoints/korean_1b_sft/checkpoint-0005000,dtype=bfloat16 \
  --tasks ko_winogrande \
  --device cuda:0 \
  --output_path ./eval/results/sft_5k_ko_winogrande.json
```

---

*이 문서는 평가 결과에 따라 업데이트 예정.*  
*다음 업데이트: Phase 1 평가 완료 후 (예상: D+1~2)*
