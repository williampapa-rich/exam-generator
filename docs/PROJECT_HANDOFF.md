# exam-generator 프로젝트 인수인계 문서

작성일: 2026-04-30
상태: **개발 중단 / 보류** (신규 프로젝트 착수로 인한 일시 중지)

이 문서는 현재까지의 트러블/해결 내역, 미해결 TODO, 그리고 **신규 프로젝트(구문분석 에디터 + 문제 자동 생성 통합 앱)** 로 가져갈 핵심 노하우를 정리한 인수인계 자료다.

---

## 1. 프로젝트 개요

### 목표
한국 수능 영어(18~45번) 자동 출제기. FastAPI + multi-LLM(Gemini/OpenAI/Anthropic) provider native structured output 으로 Question Pydantic 모델을 받아 hwpx 시험지로 렌더.

### 지원 유형 (24개)
목적(18), 심경(19), 주장(20), 밑줄함의(21), 요지(22), 주제(23), 제목(24), 인물일치(26), 안내문(27/28), 어법(29), 어휘(30), 빈칸-구(31), 빈칸-절(32~34), 무관문장(35), 순서배열(36/37), 문장삽입(38/39), 요약문(40), 장문(41-42), 장문독해(43-45). 도표(25)/듣기(1~17) 미지원.

### 기술 스택
- 백엔드: FastAPI + Pydantic 2 + asyncio
- LLM: Gemini 2.5 Flash (기본) / OpenAI / Anthropic — `app/llm/base.py` get_client 로 provider 분기
- 렌더러: hwpx native (template_injector + builder), underline charPr 분리 처리
- 검증: validators.py — 유형별 마커/길이/구조 검증 후 최대 3회 재시도, 실패 시 fallback placeholder

---

## 2. 현재 성능 (시험지-19 ~ 시험지-20 기준)

### 장문독해(43-45) 성공률 변천

| 시점 | 성공률 | 핵심 변경 |
|---|---|---|
| Baseline | 10% | Phase 1 이전 |
| Phase 1 적용 후 | 50% | 길이 통제 (scope 기반) + temperature 반전 |
| Hotfix 12 (시스템 후처리) | 50% → 100% | 라벨 마커 자동 부착 |
| Hotfix 15 이후 | 60% (1회 호출 시도) | 길이 상한 1.15→1.25 완화 |
| Hotfix 18 이후 | 100% (시험지 출력) | 셔플 + Y 명시 호칭 |

### 남은 결함 — 콘텐츠 품질

시스템 검증 가능 결함은 거의 차단됐으나, **콘텐츠 품질**이 여전히 문제:
- **plot 수렴**: 3문제 모두 "young X / Dr. Aris / mentor wisdom / breakthrough" 동일 narrative (시험지-20)
- **45번 일치 문제 정답 부재 가능**: 5개 보기 모두 본문과 일치 (시험지-17 7번)
- **(a) himself 비문**: 시험지-17 1번/4번 (Hotfix 17 차단)

이는 **Gemini 2.5 Flash 의 강한 modal pattern** 때문 — 모델 prior 의 한계.

---

## 3. Hotfix 1~20 트러블슈팅 이력

### 결함별 차단 메커니즘 (★★★★★ 패턴이 효과적)

| 결함 | Hotfix | 메커니즘 | 신뢰도 |
|---|---|---|---|
| 라벨 인접 `_` (`_(a) _word_`) | 1, 12 | validator 정규식 + 시스템 후처리 | ★★★★★ |
| 라벨 자체 밑줄 (`_(a)_`) | 1 | validator | ★★★★★ |
| 4:1 분포 위반 (5:0/3:2) | 9, 10 | assignments Counter 직접 비교 | ★★★★★ |
| answer 미스매치 | 9 | assignments index 검증 | ★★★★★ |
| 본문 `_word_` 누락 | 12 | **시스템 후처리** (가장 큰 돌파구) | ★★★★★ |
| 라벨 등장 순서 위반 | 4 | validator 정규식 | ★★★★ |
| Y 라벨 직후 대명사 | 13 | validator | ★★★★ |
| 길이 위반 | 15 | 상한 1.15→1.25 완화 | ★★★★ |
| (f)~(z) 추가 라벨 | 16 | validator | ★★★★★ |
| 재귀대명사 (`(a) himself`) | 17-1 | validator | ★★★★★ |
| 정답 ① B-D-C 편향 | 18-2 | **시스템 셔플 후처리** | ★★★★★ |
| `his mentor` 모호 패턴 | 18-1 | validator | ★★★★ |
| 라벨 중복 (`(a)` 두 번) | 19 | validator | ★★★★★ |
| 단락 라벨 (A)~(D) 노출 | 20 | validator | ★★★★★ |

### 우리가 빠진 함정 (반드시 회피)

신규 프로젝트에서도 동일 함정 위험. 5원칙:

1. **❌ instruction 강화 루프** (Hotfix 6, 11)
   - 결함 보고 더 강한 명령형/예시 추가 → **인지 부담 폭증, 다른 결함 폭증**
   - Hotfix 11 예: 본문 예시 강화로 0/15 퇴보

2. **❌ 자가검증 (모델 honesty 의존)** (Hotfix 7)
   - `referent_check: Literal["MATCHES_PLAN", "MISMATCH_REWRITE"]` 추가
   - 결과: 모델이 무지성 OK 찍음. naturalness_check 도 39/39 OK로 거짓.
   - **교훈**: 자가검증 필드 무용. 시스템 검증 가능한 데이터만 schema에 두기.

3. **❌ 사전 commit 신뢰** (Hotfix 5, 9)
   - plan 단계에 분포/계획 commit 시키면 본문이 따라올 거라 가정
   - 결과: plan은 잘 채우지만 본문은 따로 놂.
   - **교훈**: structured output 필드 분리해도 일관성 보장 안 됨. **단일 진실 원천** (Hotfix 10).

4. **❌ 다양화 강제** (Hotfix 7-3 = answer_diversity_directive)
   - 정답 무작위 강제 + 본문 흐름 강제 명령
   - 결과: 9/9 실패. 즉시 비활성.
   - **교훈**: 인지 부담은 비선형. 한 번에 처리 가능한 강제 조건 수에 한계.

5. **검증 메커니즘 신뢰도 분류**

   | 종류 | 신뢰도 | 예시 |
   |---|---|---|
   | 시스템 후처리 + 검증 | ★★★★★ | Hotfix 12 라벨 자동 부착, Hotfix 18-2 셔플 |
   | 데이터 구조 검증 | ★★★★★ | assignments Counter |
   | 본문 정규식 검증 | ★★★★ | 라벨 순서, Y 대명사 |
   | 모델 자가검증 필드 | ★★ | naturalness_check ← **신뢰 X** |
   | 프롬프트 instruction | ★ | 인지 부담 비례하여 약화 |

### 가장 큰 돌파구 — Hotfix 12 (시스템 후처리)

**문제**: LLM이 본문 안에 `(a) _word_` 마커를 일관되게 누락 (9/9 실패)

**시도했던 실패 접근**:
- Hotfix 11: instruction에 ✓/❌ 예시 4가지 강화 → 더 악화
- 모델 prior에 맞서는 instruction 강화 루프

**해결**: **모델은 평문만 박고, 시스템이 정규식으로 underline 자동 부착**
```
LLM 출력:    (a) himself, lost in thought.
시스템 후처리: (a) _himself_, lost in thought.   ← 정규식 변환
```

**원칙**: **모델 prior 와 싸우지 말고 우회하라**. 모델이 잘 못하는 일을 시스템이 결정적으로 처리.

---

## 4. 신규 프로젝트로 가져갈 핵심 노하우

### 4.1. structured output 설계 원칙

**필드 순서가 토큰 생성 순서**:
- Pydantic 모델 정의 순서대로 LLM이 생성
- "사전 계획 commit"이 효과적인 경우: scope 정의처럼 본문보다 짧고 결정적인 메타데이터
- "사후 자가검증"이 효과적인 경우: 거의 없음 — 모델 honesty 신뢰 X

**단일 진실 원천**:
- 같은 정보를 두 필드로 받지 말 것 (Hotfix 5 distribution + assignments 중복 → 거짓말 발생)
- assignments 같은 검증 가능한 list 하나가 plan + check 보다 강함

**필드 추가 결정 기준**:
- 시스템이 그 필드로 검증 가능한가? → ✓ 추가
- 모델이 정직하게 채울 수 있는 검증인가? → ❌ 추가 X (자가검증 함정)

### 4.2. 프롬프트 설계 원칙

**총 길이 ≤ ~6000자 권장**:
- Gemini 2.5 Flash 기준 system + user 6000자 넘으면 instruction following 약화 + max_output_tokens 잘림 위험
- system은 공통 + 짧게, user에 유형별 특화 instruction

**Anchor 효과 차단**:
- 길이 가이드 노출 시 모델이 끝값으로 수렴 (예: "400~500자" → 500자 끝점)
- 정성적 표현 (`'균형있게'`)이 수치 anchor 보다 안전

**최근 N건 회피 (narrative diversity)**:
- 같은 유형 최근 success 호출의 scope를 user에 노출 + "이와 다른 narrative로"
- 효과 약함 — 직업만 바뀌고 plot 같음 (시험지-20)
- 모델 교체나 plot 카탈로그 강제 필요 (보류 중)

### 4.3. 검증 시스템 설계

**다층 검증** — 가까운 원인부터 reject:
1. 구조 검증 (개수, 형식)
2. 마커 형식 검증 (라벨, 밑줄)
3. 의미 검증 (라벨 등장 순서, 분포, 정답 일치)
4. 자가검증 (낮은 우선순위)

**reject 메시지 = 재시도 hint**:
- 모호한 사유보다 구체적 원인 지목
- 예: "본문 길이 X자" → 막연 / "단락 비대 (B)단락 700자" → 명확
- 다만 retry hint도 모델 prior에 약함 — 시스템 후처리가 우선

**idempotent**:
- 후처리는 반복 실행해도 같은 결과 (Hotfix 12 negative lookahead)
- 셔플도 라벨 매핑 후 재셔플 안전

### 4.4. multi-provider 추상화

`app/llm/base.py` 의 `LLMClient` Protocol:
```python
async def generate_json(
    *, system: str, user: str, schema: type[T],
    max_tokens: int = 8000, temperature: float = 0.9,
) -> tuple[T, UsageInfo]: ...
```

provider별 native structured output:
- Gemini: `response_schema=` (Pydantic 직접 전달)
- OpenAI: `response_format={"type": "json_schema"}`
- Anthropic: tool_use 강제 + `input_schema=model_json_schema`

`UsageInfo` 통일 (model, prompt_tokens, completion_tokens) → 비용 계산/로깅 일관.

### 4.5. 비용 최적화

토큰당 가격 (1M token, USD):
| 모델 | input | output | 1문제 (KRW) |
|---|---|---|---|
| Gemini 2.5 Flash | $0.30 | $2.50 | ~5원 |
| Gemini 2.5 Pro | $1.25 | $10 | ~19원 |
| GPT-4o | $2.50 | $10 | ~23원 |
| Claude Sonnet 4.6 | $3 | $15 | ~32원 |
| Claude Opus 4.7 | $15 | $75 | ~160원 |

평균 in 2247 / out 1073 tokens 기준. 재시도 포함 실측 1.45회 → 약 2배 비용.

**유형별 라우팅** 가능: 장문독해는 Sonnet, 일반 유형은 Flash 같은 분기.

### 4.6. 로깅 / 분석

`tools/stats/usage_log.jsonl` 한 줄 = 한 호출:
- 토큰/비용/시간 + plan_topic_scope/assignments/answer/temperature/fail_reason
- 사후 분석:
  ```python
  rows = [json.loads(l) for l in open('usage_log.jsonl')]
  Counter(r['fail_reason'][:30] for r in rows if not r['success'])  # 결함 분포
  Counter(r['answer'] for r in rows if r['success'])  # 정답 편향
  ```

DB 마이그레이션은 1시간 작업. 초기는 jsonl 충분.

---

## 5. 미해결 TODO

### 즉시 진행 가능 (코드 작업)

1. **모델 교체 시도 (Claude Sonnet 4.6)**
   - `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` 만 설정
   - 장문독해(43-45)에서 narrative 다양성 개선 기대
   - 비용 5~6배 증가 (시험지 1세트당 약 6,700원)
   - 또는 유형별 라우팅 (장문만 Sonnet, 일반은 Flash)

2. **(B)단락 자연 흐름 재배치 검증**
   - Hotfix 18-2 셔플은 sub_passages만 섞음
   - LLM이 자연 흐름과 다르게 sub_passages를 박을 수도 있음 (예: [C', B', D'] 순으로 출력)
   - 셔플 전 자연 흐름 검증 — naturalness_check 비슷하지만 더 좁은 범위

3. **본문 cross-reference 차단**
   - "Earlier in (B)..." 같이 단락 라벨 인용 패턴 발견 시 reject
   - Hotfix 20에 일부 포함됨 (`(B)` 매칭) 하지만 미세 조정 필요

### 콘텐츠 품질 (모델 차원 한계)

4. **plot 다양화** (시험지-20 결함)
   - 현재: "young X / Dr. Aris / breakthrough" 패턴 수렴
   - 옵션 A (instruction 카탈로그) — 함정 위험 ⚠️
   - 옵션 B (모델 교체) — TODO 1번
   - 옵션 C (`reference_text` 적극 활용 — 이미 prompt_for_type 시그니처에 있으나 GUI에서 미사용)

5. **45번 일치 문제 정답 부재 검증**
   - 5개 보기 모두 본문과 일치하면 정답 없음 (시험지-17 7번)
   - LLM judge 또는 별도 검증 모델 필요
   - 정규식 못 잡음. 우선순위 낮음.

6. **재귀대명사 검증 (Hotfix 17-1) 위양성 점검**
   - "(b) himself, lost in thought" 같이 동격 부사구로 자연스러운 케이스 — 현재 무조건 reject
   - 평가원 실제 출제에 재귀대명사 단독 라벨 케이스 있는지 확인 필요

### 인프라

7. **테스트 — pytest-asyncio 미설치**
   - 비동기 테스트 4개 (`test_generate_one_*`, `test_generate_all_parallel`) deselect 중
   - `pip install pytest-asyncio` + `asyncio_mode = "auto"` 추가하면 즉시 동작

8. **usage_log.jsonl 누적 정리**
   - Phase 1 이전 baseline 데이터 + 이후 데이터 혼재
   - 분석 스크립트가 모든 행을 한 번에 다룸 (속도 OK, 가독성 OK)
   - DB 이전 시 `phase: int` 컬럼 추가 검토

### Phase 2 (보류)

9. **SCOPE_GUIDE 도입** — 상습 위반 유형(요지/빈칸-절/순서배열)에 단락별 기능 명시
   - 인지 부담 폭증 위험 ⚠️
   - 우선 모델 교체 시도 후 결정

10. **SYSTEM_PROMPT 슬림화** — 현재 system 2592자, 마커 규약 섹션을 user로 이동
    - lost-in-the-middle 완화 기대
    - 부분 적용됨 (Hotfix 1.2)

---

## 6. 신규 프로젝트(구문분석 에디터 + 문제 자동 생성) 권장 사항

### 재사용 가능한 자산

**즉시 재사용**:
- `app/llm/base.py` — multi-provider Protocol + UsageInfo
- `app/llm/usage_log.py` — jsonl 로깅
- `app/llm/pricing.py` — 토큰 → 비용 환산
- `shared/schemas/question.py` 의 `QuestionPlan`, `Question` 패턴 (구조 일부)
- `validators.py` 의 정규식 라이브러리 (라벨/마커 처리)

**부분 재사용**:
- `generator.py` 의 retry loop + temperature 스케줄
- `prompts.py` 의 `_recent_scopes_directive` (narrative 회피)
- `app/renderer/template_injector.py` (hwpx 렌더 패턴 — 신규 프로젝트가 다른 출력 형식이면 무관)

### 적용 권장 패턴

1. **첫날부터 시스템 후처리 우선** — instruction 강화는 마지막 수단
2. **단일 진실 원천 schema** — 같은 정보 두 곳에 저장 금지
3. **자가검증 필드 도입 보류** — 데이터로 효과 입증 후 결정
4. **로깅부터 — jsonl 한 줄 한 호출, 모든 메타데이터 기록**
5. **테스트 13개 + 회귀 매번 실행** — sync 단위 테스트가 하드 가드

### 구문분석 에디터 — 새로운 도전

수능 출제기와 다른 점:
- **출력보다 input 분석**: 영어 문장의 syntactic tree, 어휘 hint, 구조 표시
- **Real-time interactive**: 사용자 편집 중 즉시 분석 vs 일괄 생성
- **부분 검증** vs 전체 검증: 한 문장만 분석해도 결과 신뢰

→ structured output 모델은 동일하지만 **속도/streaming** 가 더 중요. Anthropic streaming + 부분 결과 표시 검토.

문제 자동 생성 통합 시:
- 구문 분석 → 어휘/구조 hint 추출 → exam-generator schema의 `reference_text` 로 주입
- 이러면 plot 수렴 문제 (TODO 4)도 부분 해결 — 구체 reference 주면 모델이 다양화

---

## 7. 환경 / 실행 정보

### 실행
```bash
cd /Users/william/workspace/exam-generator
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 환경변수
- `LLM_PROVIDER`: gemini (기본) / openai / anthropic
- `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
- `USAGE_LOG_PATH`: 기본 `tools/stats/usage_log.jsonl`

### 테스트
```bash
.venv/bin/python -m pytest tests/test_llm_adapters.py -q \
  --deselect tests/test_llm_adapters.py::test_generate_one_success \
  --deselect tests/test_llm_adapters.py::test_generate_one_overrides_meta_from_config \
  --deselect tests/test_llm_adapters.py::test_generate_one_falls_back_on_exception \
  --deselect tests/test_llm_adapters.py::test_generate_all_parallel
# 13 passed (sync only)
```

### Git 상태
- 마지막 커밋: `45d3ea3` (Phase 1 + hotfix 4건)
- Hotfix 5~20 미커밋 (작업 중단 시점에 commit 필요)

---

## 8. 마무리 코멘트

이 프로젝트는 **LLM 의 강한 prior 와 어떻게 협상할 것인가** 의 케이스 스터디다. 결론:

- **인지 부담 < 시스템 후처리**: 모델에게 더 시키기보다 시스템이 받아 처리
- **단일 진실 원천 < 다중 검증 필드**: schema 단순화가 신뢰성 ↑
- **콘텐츠 품질은 모델 한계**: 검증으로 잡을 수 있는 결함과 못 잡는 결함을 구분

신규 프로젝트가 같은 함정에 빠지지 않도록, 이 문서의 **3.우리가 빠진 함정 5원칙**을 첫날부터 인쇄해서 옆에 두고 시작할 것을 권장.
