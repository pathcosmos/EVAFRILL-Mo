# EVAFRILL-Mo 3B — 종합 평가 보고서

- **평가 일시**: 2026-03-10 00:12
- **체크포인트**: `checkpoint-0319772`
- **총 소요 시간**: 84.0분

---

## 1. Executive Summary

### PPL (주요 셋)

| 데이터셋 | PPL |
|---------|-----|
| 3b | 5.8402 |

### 생성 품질 요약

| 설정 | 평균 3-gram 반복률 | EOS 종료율 |
|------|-------------------|-----------|
| greedy | 87.03% | 0% |
| temp=0.7 rep=1.2 | 2.65% | 0% |

### Calibration

| Top-1 | Top-5 | Top-10 |
|-------|-------|--------|
| 63.63% | 78.12% | 82.99% |

### lm-eval 벤치마크

| 태스크 | Accuracy | 랜덤 기준 |
|--------|----------|----------|
| arc_challenge | 0.1920 | 0.25 |
| arc_easy | 0.3413 | 0.25 |
| belebele_kor_Hang | 0.2311 | 0.25 |
| global_mmlu_full_ko | 0.2310 | 0.25 |
| global_mmlu_full_ko_humanities | 0.2408 | ? |
| global_mmlu_full_ko_formal_logic | 0.2540 | ? |
| global_mmlu_full_ko_high_school_european_history | 0.2303 | ? |
| global_mmlu_full_ko_high_school_us_history | 0.2598 | ? |
| global_mmlu_full_ko_high_school_world_history | 0.2658 | ? |
| global_mmlu_full_ko_international_law | 0.2397 | ? |
| global_mmlu_full_ko_jurisprudence | 0.2593 | ? |
| global_mmlu_full_ko_logical_fallacies | 0.2209 | ? |
| global_mmlu_full_ko_moral_disputes | 0.2428 | ? |
| global_mmlu_full_ko_moral_scenarios | 0.2380 | ? |
| global_mmlu_full_ko_philosophy | 0.1929 | ? |
| global_mmlu_full_ko_prehistory | 0.2099 | ? |
| global_mmlu_full_ko_professional_law | 0.2451 | ? |
| global_mmlu_full_ko_world_religions | 0.3099 | ? |
| global_mmlu_full_ko_other | 0.2359 | ? |
| global_mmlu_full_ko_business_ethics | 0.2900 | ? |
| global_mmlu_full_ko_clinical_knowledge | 0.2151 | ? |
| global_mmlu_full_ko_college_medicine | 0.1965 | ? |
| global_mmlu_full_ko_global_facts | 0.2200 | ? |
| global_mmlu_full_ko_human_aging | 0.3004 | ? |
| global_mmlu_full_ko_management | 0.1942 | ? |
| global_mmlu_full_ko_marketing | 0.2863 | ? |
| global_mmlu_full_ko_medical_genetics | 0.2800 | ? |
| global_mmlu_full_ko_miscellaneous | 0.2248 | ? |
| global_mmlu_full_ko_nutrition | 0.2255 | ? |
| global_mmlu_full_ko_professional_accounting | 0.2411 | ? |
| global_mmlu_full_ko_professional_medicine | 0.2022 | ? |
| global_mmlu_full_ko_virology | 0.2470 | ? |
| global_mmlu_full_ko_social_sciences | 0.2158 | ? |
| global_mmlu_full_ko_econometrics | 0.2281 | ? |
| global_mmlu_full_ko_high_school_geography | 0.1667 | ? |
| global_mmlu_full_ko_high_school_government_and_politics | 0.1969 | ? |
| global_mmlu_full_ko_high_school_macroeconomics | 0.2051 | ? |
| global_mmlu_full_ko_high_school_microeconomics | 0.2059 | ? |
| global_mmlu_full_ko_high_school_psychology | 0.2000 | ? |
| global_mmlu_full_ko_human_sexuality | 0.2901 | ? |
| global_mmlu_full_ko_professional_psychology | 0.2484 | ? |
| global_mmlu_full_ko_public_relations | 0.1818 | ? |
| global_mmlu_full_ko_security_studies | 0.1796 | ? |
| global_mmlu_full_ko_sociology | 0.2338 | ? |
| global_mmlu_full_ko_us_foreign_policy | 0.2800 | ? |
| global_mmlu_full_ko_stem | 0.2265 | ? |
| global_mmlu_full_ko_abstract_algebra | 0.2200 | ? |
| global_mmlu_full_ko_anatomy | 0.2148 | ? |
| global_mmlu_full_ko_astronomy | 0.1842 | ? |
| global_mmlu_full_ko_college_biology | 0.2778 | ? |
| global_mmlu_full_ko_college_chemistry | 0.1900 | ? |
| global_mmlu_full_ko_college_computer_science | 0.2900 | ? |
| global_mmlu_full_ko_college_mathematics | 0.2200 | ? |
| global_mmlu_full_ko_college_physics | 0.2255 | ? |
| global_mmlu_full_ko_computer_security | 0.2500 | ? |
| global_mmlu_full_ko_conceptual_physics | 0.2511 | ? |
| global_mmlu_full_ko_electrical_engineering | 0.2552 | ? |
| global_mmlu_full_ko_elementary_mathematics | 0.2196 | ? |
| global_mmlu_full_ko_high_school_biology | 0.1968 | ? |
| global_mmlu_full_ko_high_school_chemistry | 0.2167 | ? |
| global_mmlu_full_ko_high_school_computer_science | 0.2800 | ? |
| global_mmlu_full_ko_high_school_mathematics | 0.2148 | ? |
| global_mmlu_full_ko_high_school_physics | 0.2384 | ? |
| global_mmlu_full_ko_high_school_statistics | 0.1713 | ? |
| global_mmlu_full_ko_machine_learning | 0.3036 | ? |
| hellaswag | 0.2866 | 0.25 |

## 2. 생성 샘플 (Greedy)

**[대한민국의 수도는]**
> 서울특별시이고, 그 외 지역은 광역시로 분류된다.대한민국의 수도는 서울특별시이고, 그 외 지역은 광역시로 분류된다.대한민국의 수도는 서울특별시이고, 그 외 지역은 광역시로 분류된다.대한민국의 수도는 서울특별시이고, 그 외 지역은 광역시로 분류된다.대한민국의 수도는 서울특별시이고, 그 외 지역은 광역시로 분류된다.대한민국의 수도는 서울특별시이고, 그 외 지역
> *EOS=False, 3gram_rep=94.09%, tokens=256*

**[인공지능이란]**
> 무엇인가?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 발전은 어디까지?- 인공지능의 
> *EOS=False, 3gram_rep=96.46%, tokens=256*

**[한국의 전통 음식 중에서]**
> 가장 유명한 것이 바로 김치다. 김치는 한국인의 밥상에서 빼놓을 수 없는 음식으로, 김치를 담그는 데는 많은 시간과 정성이 들어간다. 김치를 담그는 데는 배추를 비롯한 여러 채소와 고춧가루, 소금, 물, 김치국물, 김치통, 김치냉장고, 김치냉장고용 김치통, 김치용기, 김치용기용 용기, 김치용기용 용기용기, 김치용기용 용기용기, 김치용기용 용기용기, 김치용기
> *EOS=False, 3gram_rep=71.65%, tokens=256*

**[지구 온난화의 주요 원인은]**
> 이산화탄소의 증가라고 한다.이산화탄소는 대기 중의 이산화탄소 농도가 증가함에 따라 지구 온난화가 가속되는 현상을 말한다. 이산화탄소는 대기 중의 이산화탄소 농도가 증가함에 따라 지구 온난화가 가속되는 현상을 말한다. 이산화탄소는 대기 중의 이산화탄소 농도가 증가함에 따라 지구 온난화가 가속되는 현상을 말한다. 이산화탄소는 대기 중의 이산화탄소 농도가 증가함
> *EOS=False, 3gram_rep=87.40%, tokens=256*

**[프로그래밍을 배우려면]**
> , 프로그래밍 언어를 배워야 합니다.이제, 여러분은 자바스크립트에 대한 기본 지식을 가지고 있습니다.이제, 여러분은 자바스크립트에 대한 기본 지식을 가지고 있습니다.이제, 여러분은 자바스크립트에 대한 기본 지식을 가지고 있습니다.이제, 여러분은 자바스크립트에 대한 기본 지식을 가지고 있습니다.이제, 여러분은 자바스크립트에 대한 기본 지식을 가지고 있습니다.
> *EOS=False, 3gram_rep=91.34%, tokens=256*

**[조선시대에는]**
> ‘이방(異邦)’이라는 말이 없었다. 이방은 ‘이방인’의 줄임말이다. 이방은 ‘이방인’의 반대말이다. 이방은 ‘이방인’의 반대말이다. 이방은 ‘이방인’의 반대말이다. 이방은 ‘이방인’의 반대말이다. 이방은 ‘이방인’의 반대말이다. 이방은 ‘이방인’의 반대말이다. 이방은 ‘이방인’의 반대말이다. 이방은 ‘이방인’의 반대말이다. 이방은 ‘이방인’의 반대말이다. 
> *EOS=False, 3gram_rep=90.16%, tokens=256*

**[물리학에서 에너지란]**
> 물질의 운동에너지를 의미한다. 에너지는 물질의 운동에너지를 의미한다. 에너지는 물질의 운동에너지를 의미한다. 에너지는 물질의 운동에너지를 의미한다. 에너지는 물질의 운동에너지를 의미한다. 에너지는 물질의 운동에너지를 의미한다. 에너지는 물질의 운동에너지를 의미한다. 에너지는 물질의 운동에너지를 의미한다. 에너지는 물질의 운동에너지를 의미한다. 에너지는 물질
> *EOS=False, 3gram_rep=96.46%, tokens=256*

**[한국어는 세계에서]**
> 이 다만s 스카는 세계에서이 다만s goods는 display 사제 기사단이 다만s/11는 난다 아빠 되어이 다만s 2023 E는 개인의 While이 다만s 없어진 E는 이루어진 major been이 다만s 2023 E는 그 by 3이 다만s Sub E는 불길이 다만s 없어진 E는 이루어진 major been이 다만s 시청자들은는 보통 서로이 다만s 시청자
> *EOS=False, 3gram_rep=78.35%, tokens=256*

**[경제 성장을 위해서는]**
> 노동시장의 유연성이 확보되어야 한다. 노동시장의 유연성 확보를 위해서는 노동시장의 유연성 확보를 위한 노동시장의 유연화(flexible)가 필요하다. 노동시장의 유연성 확보를 위해서는 노동시장의 유연화(flexible)가 필요하다. 노동시장의 유연성 확보를 위해서는 노동시장의 유연화(flexible)가 필요하다. 노동시장의 유연성 확보를 위해서는 노동시장의
> *EOS=False, 3gram_rep=89.37%, tokens=256*

**[우주 탐사의 역사를 보면]**
> , 우주탐사의 역사는 인류의 역사와 함께 해왔다.우주탐사의 역사는 인류의 역사와 함께 해왔다.우주탐사의 역사는 인류의 역사와 함께 해왔다.우주탐사의 역사는 인류의 역사와 함께 해왔다.우주탐사의 역사는 인류의 역사와 함께 해왔다.우주탐사의 역사는 인류의 역사와 함께 해왔다.우주탐사의 역사는 인류의 역사와 함께 해왔다.우주탐사의 역사는 인류의 역사와 함께 해왔
> *EOS=False, 3gram_rep=95.28%, tokens=256*

**[머신러닝과 딥러닝의 차이는]**
> 무엇일까요?딥러닝은 기계가 스스로 학습하는 것을 말합니다. 기계가 스스로 학습하는 것을 딥러닝이라고 합니다. 딥러닝은 기계가 스스로 학습하는 것을 말합니다. 기계가 스스로 학습하는 것을 딥러닝이라고 합니다. 딥러닝은 기계가 스스로 학습하는 것을 말합니다. 기계가 스스로 학습하는 것을 딥러닝이라고 합니다. 딥러닝은 기계가 스스로 학습하는 것을 말합니다. 기계
> *EOS=False, 3gram_rep=89.76%, tokens=256*

**[한국 문학의 대표적인 작품으로는]**
> 이문열의 『무궁화꽃이 피었습니다』와 김훈 작가의 『칼의 노래』가 있다.이문열의 『무궁화꽃이 피었습니다』는 1980년대의 암울한 시대상을 배경으로 한 소설로, 1980년대의 암울한 시대상을 배경으로 한 소설로, 1980년대의 암울한 시대상을 배경으로 한 소설로, 1980년대의 암울한 시대상을 배경으로 한 소설로, 1980년대의 암울한 시대상을 배경으로 한 소
> *EOS=False, 3gram_rep=83.86%, tokens=256*

**[양자 컴퓨터란]**
> ?[기고] 인공지능(AI)과 머신러닝의 미래[기고] 인공지능(AI)과 머신러닝의 미래[기고] 인공지능(AI)과 머신러닝의 미래[기고] 인공지능(AI)과 머신러닝의 미래[기고] 인공지능(AI)과 머신러닝의 미래[기고] 인공지능(AI)과 머신러닝의 미래[기고] 인공지능(AI)과 머신러닝의 미래[기고] 인공지능(AI)과 머신러닝의 미래[기고] 인공지능(AI)과 
> *EOS=False, 3gram_rep=93.70%, tokens=256*

**[건강한 식습관을 위해서는]**
> 하루 세끼를 꼭 챙겨 먹어야 한다.식습관 개선을 위해서는 우선 아침식사를 거르지 않는 것이 중요하다. 아침식사를 거르면 소화가 잘 되지 않아 소화불량에 걸리기 쉽다. 또 아침식사를 거르면 소화가 잘 되지 않아 배탈이 날 수 있다. 따라서 아침식사를 거르지 않고 먹는 것이 좋다.식습관 개선을 위해서는 간식을 먹는 것도 중요하다. 간식을 먹으면 소화가 잘 되지
> *EOS=False, 3gram_rep=62.60%, tokens=256*

**[세계 2차 대전 이후]**
> , 미국을 중심으로 한 세계 질서의 재편을 의미한다.1945년 8월 15일 일본의 항복으로 제2차 세계 대전은 막을 내렸다. 1945년 8월 15일 일본의 항복으로 제2차 세계 대전은 막을 내렸다. 제2차 세계 대전 이후, 미국을 중심으로 한 세계 질서의 재편을 의미한다.1945년 8월 15일 일본의 항복으로 제2차 세계 대전은 막을 내렸다. 제2차 세계 
> *EOS=False, 3gram_rep=85.04%, tokens=256*
