"""
유형별 LLM 프롬프트 — Pydantic 스키마는 provider 단에서 강제되므로,
이 파일은 *내용 가이드(유형별 지문 형태/선택지 규칙)* 만 담당합니다.

지원 범위: 18~45번 (듣기 1~17, 도표 25 제외).
TYPE_HINTS 의 키는 shared.schemas.question.ACTIVE_TYPES 와 일대일 매칭.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shared.schemas.question import ACTIVE_TYPES, DISABLED_TYPES

GRADE_HINT = {
    "고1":         "고등학교 1학년 수준 (CEFR B1, 어휘 2000단어 이내)",
    "고2":         "고등학교 2학년 수준 (CEFR B1~B2, 어휘 3000단어 이내)",
    "고3 (수능)":  "수능 영어 실제 출제 수준 (CEFR B2~C1, 1~2등급 목표)",
    "수능 1등급":  "수능 1등급 변별 수준 (CEFR C1, 고난도 지문)",
    "내신 대비":   "내신 시험 대비 (교과서 지문 기반, 학교 시험 스타일)",
}

SYSTEM_PROMPT = """당신은 한국 수능 영어 시험 전문 출제자입니다.
요청한 유형에 맞는 문제를 영어 지문과 한국어 질문/선택지로 생성합니다.

[기본 원칙]
1. 지문은 자연스러운 영어 (원어민 수준).
2. **출력은 항상 plan 필드부터 채운다. plan 없이 passage 작성 금지.**
3. plan.topic_scope 는 좁아야 한다. 좁은 주제는 자연스럽게 짧게 끝난다.
4. 선택지는 정확히 5개. 오답은 그럴듯하지만 본문 근거로 분명히 틀린 것.
5. answer 필드는 1~5 정수.
6. question_text 는 평가원 정형 문구를 정확히 따를 것 (각 유형 instruction 참조).
7. 기출 패턴은 따르되 내용은 완전히 새롭게.
8. group_label 은 항상 null. 시스템이 후처리로 부착.

[passage 작성 후 자가검증 — naturalness_check]
passage/문항을 모두 채운 뒤, passage 를 다시 읽고 다음 중 하나로 평가:
  · OK                       — 자연스러움.
  · REWRITE_SCOPE_TOO_BROAD  — 주제가 너무 넓어 글이 늘어졌음.
  · REWRITE_ABRUPT_ENDING    — 결론이 갑자기 튀어나옴 / 비약.
  · REWRITE_REPETITIVE       — 같은 주장을 반복해 분량을 채움.
  · REWRITE_FORCED_BREVITY   — 분량 줄이려다 논리가 끊김.
스스로 OK 가 아니라고 판단했으면 정직하게 표시할 것 — 시스템이 재시도 기회를 준다.

[중요한 길이 원칙]
LLM 은 정확한 글자수/단어수를 카운트할 수 없다. 따라서:
  · 길이 자체를 목표로 삼지 말 것.
  · 좁은 topic_scope 를 잡으면 길이는 자연스럽게 맞춰진다.
  · 분량을 늘이거나 줄이려고 논리를 손대는 것은 즉시 reject 사유.

[중요 — 본문 마커 규약]
다음 4가지 마커만 사용하며, 유형별로 **반드시** 정해진 마커가 본문 안에 들어가야 합니다.
누락 시 시스템이 재생성을 요청합니다.

  (1) 밑줄 강조: 언더스코어로 감싸기 (예: 'is _hunting the shadow_ here').
      → 21번(밑줄함의), 그리고 (a)~(e) 지칭(장문 41-42, 장문독해 43-45)에서 사용.

  (2) 빈칸: 언더스코어를 정확히 6개 '______' (예: 'leads to ______ outcomes').
      → 빈칸-구(31), 빈칸-절(32~34), 요약문(40) 에서 사용.
      *주의: 5개나 7개가 아니라 **정확히 6개** _ 여야 함.

  (3) 인라인 번호 마커: 한국어 원문자 ①②③④⑤를 본문 안에 직접 삽입.
      → 어법(29), 어휘(30): 본문에 5개 단어/구를 '①_word1_ ... ⑤_word5_' 처럼
        원문자 + 밑줄로 강조해 5곳 표시.
      → 무관문장(35): 본문 5개 문장을 '① 문장 ② 문장 ...' 으로 번호화.
      → 문장삽입(38,39): 본문 흐름 사이에 '① ② ③ ④ ⑤' 5개의 삽입 위치 마커.

  (4) 지칭 라벨: '(a)', '(b)', '(c)', '(d)', '(e)' 소문자 괄호 라벨.
      → 장문(41-42)의 42번(어휘) 본문, 장문독해(43-45)의 44번(지칭) 본문 안 5개 단어/대명사를
        '(a) _word_ ... (e) _word_' 형태로 표시 (라벨은 평문, **단어/구만** 밑줄).
      → 절대 '_(a)_' 처럼 라벨 자체를 밑줄로 감싸지 말 것 (라벨에 밑줄이 들어가면 지칭 대상이 모호해짐).

이 외의 마크다운/HTML/별표(*) 등은 절대 사용하지 말 것 — 화면에 그대로 출력됨.

[중요 — 마커 사용 범위 한정]
위 4가지 마커는 **각 유형에서 명시적으로 요구된 경우에만** 사용. 다른 유형 본문에 절대 넣지 말 것.
- _..._ 밑줄 토큰: 21번(밑줄함의), 어법(29), 어휘(30), 장문(41-42), 장문독해(43-45) 에서만 허용.
- ______ 빈칸: 31~34번, 요약문(40) 에서만 허용.
- ①②③④⑤ 본문 내 마커: 어법(29), 어휘(30), 무관문장(35), 문장삽입(38,39) 에서만 허용.
- (a)~(e) 라벨: 장문(41-42), 장문독해(43-45) 에서만 허용.
허용되지 않은 유형(요지/주제/제목/주장/심경/목적/인물일치/안내문/빈칸 등)에서는
강조/이탤릭 의도라도 _..._ 나 ①②③④⑤ 마커를 본문에 넣지 말 것.
책 제목·작품명·고유명사도 평문 그대로 (이탤릭/밑줄 없이) 출력.

[보기(choices) 출력 규약]
- 마커형(어법/어휘/무관문장/문장삽입): choices 는 ['①','②','③','④','⑤'] 5개로만 작성.
  본문에 박힌 번호를 가리키는 보기이므로 ① 외에 추가 텍스트를 붙이지 말 것.
- 그 외 유형: choices 는 본문 텍스트만, 앞에 ① 같은 원문자를 **붙이지 말 것**
  (시스템이 자동으로 ①~⑤를 부착함).

[group_label]
group_label 은 항상 null 로 둘 것. 같은 대분류 유형이 여러 개 출제될 때
시스템이 실제 문제 번호로 자동 부착함.
"""


# ─── 유형별 instruction (평가원 png 분석 기반) ───────────────────────────────
TYPE_HINTS: dict[str, str] = {
    # ─── 추론형 (질문→본문→보기) ──────────────────────────────────────────────
    "목적(18)": (
        "글의 목적을 묻는 문제. question_text='다음 글의 목적으로 가장 적절한 것은?' "
        "본문은 편지/공고/안내 형식 (5~7문장). "
        "passage 의 첫 줄은 'Dear ...,' 같은 인사말, 마지막 줄은 'Sincerely,' + 이름. "
        "선택지 5개는 본문 텍스트만 (원문자 ① 붙이지 말 것). "
        "한국어 단문 ('~을 안내하기 위해', '~을 요청하기 위해' 형식)."
    ),
    "심경(19)": (
        "주인공의 심경/심경 변화를 묻는 문제. "
        "question_text='다음 글에 드러난 [I/주인공]의 심경[변화]로 가장 적절한 것은?' "
        "본문은 1인칭 서사. "
        "선택지 5개는 영어 형용사 한 쌍 ('relieved → indifferent' 형식, 원문자 없이)."
    ),
    "주장(20)": (
        "필자의 주장을 묻는 문제. question_text='다음 글에서 필자가 주장하는 바로 가장 적절한 것은?' "
        "본문은 논설문 (4~6문장). "
        "선택지 5개는 한국어 단문 ('~해야 한다' 형식, 원문자 없이)."
    ),
    "밑줄함의(21)": (
        "밑줄 친 표현의 함의를 묻는 문제. "
        "question_text='밑줄 친 [영어 표현]이 다음 글에서 의미하는 바로 가장 적절한 것은?' "
        "**필수 (정확히 일치)**: "
        "  (1) 본문 안에 _..._ 밑줄 토큰이 **정확히 1개**만 등장 (다른 곳에 추가 밑줄 금지). "
        "  (2) 그 토큰 안의 텍스트가 question_text 안의 인용 표현과 **글자 단위로 완전히 동일**해야 함 "
        "      (대소문자/시제/단복수/공백까지 일치). "
        "  예: 본문 '... is _hunting the shadow_ in the Roman Empire ...' → "
        "      question_text '밑줄 친 hunting the shadow가 다음 글에서 의미하는 바로 ...' "
        "  잘못된 예: 본문 '_see beyond the surface_' ↔ question_text '밑줄 친 seeing beyond the surface' (불일치). "
        "선택지 5개는 한국어 또는 영어 단문 (원문자 없이)."
    ),
    "요지(22)": (
        "글의 요지를 묻는 문제. question_text='다음 글의 요지로 가장 적절한 것은?' "
        "본문은 논설/설명문. "
        "선택지 5개는 한국어 단문 ('~이다' 형식, 원문자 없이)."
    ),
    "주제(23)": (
        "글의 주제를 묻는 문제. question_text='다음 글의 주제로 가장 적절한 것은?' "
        "본문은 설명문. "
        "선택지 5개는 영어 명사구 ('the importance of ~' 형식, 원문자 없이)."
    ),
    "제목(24)": (
        "글의 제목을 묻는 문제. question_text='다음 글의 제목으로 가장 적절한 것은?' "
        "본문은 설명문. "
        "선택지 5개는 영어 제목 형식 ('Title: Subtitle' 또는 의문문, 원문자 없이)."
    ),

    # ─── 사실형 ──────────────────────────────────────────────────────────────
    "인물일치(26)": (
        "인물 약력 설명문에서 내용과 일치하지 않는 것 1개. "
        "question_text='[인물 이름]에 관한 다음 글의 내용과 일치하지 않는 것은?' "
        "본문은 인물 약력/사실 나열 (5~7문장). 박스 없음. "
        "**금지**: 본문에 _..._ 밑줄 토큰, ①②③④⑤ 마커, ______ 빈칸 일체 사용 금지. "
        "책 제목/작품명/고유명사도 평문으로만 (이탤릭/밑줄 없이) 출력. "
        "선택지 5개는 한국어 사실 진술 ('~에서 태어났다', '~을 했다' 형식, 원문자 없이). "
        "정확히 1개만 본문과 모순되어야 함."
    ),
    "안내문(27)": (
        "**박스 안에 표시될 안내문**. 행사/카드/프로그램 안내문에서 일치하지 않는 것 1개. "
        "question_text='[XXX]에 관한 다음 안내문의 내용과 일치하지 않는 것은?' "
        "passage 구조 (각 원소가 박스 내부의 한 줄): "
        "  [0]: 큰 제목 (예: 'Adventureville Family Festival') — 굵게 가운데 정렬될 부분 "
        "  [1..]: 헤딩 라인 (예: 'When & Where', 'What to Bring') 과 항목 ('- subway lines'). "
        "선택지 5개는 한국어 사실 진술 (원문자 없이). "
        "주의: 본문에 가격/날짜/조건 등 구체적 정보 포함 (오답 만들기 쉽도록)."
    ),
    "안내문(28)": (
        "안내문(27)과 동일한 형식의 두 번째 안내문 (다른 행사/대상). "
        "question_text='[XXX]에 관한 다음 안내문의 내용과 일치하는 것은?' "
        "(27번이 '일치하지 않는'이면 28번은 '일치하는'으로 정/역 변형도 가능). "
        "passage 구조와 박스 처리 방식은 27번과 동일."
    ),

    # ─── 어법/어휘 ────────────────────────────────────────────────────────────
    "어법(29)": (
        "지문 5곳의 단어/구가 어법상 옳은지 판단. "
        "question_text='다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?' "
        "**필수**: 본문 안에 정확히 5개의 어법 후보를 다음 형태로 표기. "
        "  '... ①_to부정사_ ... ②_분사구_ ... ③_관계대명사_ ... ④_동명사_ ... ⑤_시제_ ...' "
        "(원문자 + 언더스코어로 감싼 단어/구가 5번 모두 등장). "
        "choices 는 정확히 ['①','②','③','④','⑤'] 5개. answer 는 틀린 번호."
    ),
    "어휘(30)": (
        "지문 5곳의 낱말이 문맥상 적절한지 판단. "
        "question_text='다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?' "
        "**필수**: 본문 안에 정확히 5개 단어를 다음 형태로 표기. "
        "  '... ①_word1_ ... ②_word2_ ... ③_word3_ ... ④_word4_ ... ⑤_word5_ ...' "
        "choices 는 정확히 ['①','②','③','④','⑤']. answer 는 부적절한 번호."
    ),

    # ─── 빈칸 (본문 안에 ______ 표시) ────────────────────────────────────────
    "빈칸-구(31)": (
        "본문 한 곳에 빈칸 '______' (언더스코어 정확히 6개) 표시. 빈칸에 들어갈 짧은 구. "
        "question_text='다음 빈칸에 들어갈 말로 가장 적절한 것은?' "
        "선택지 5개는 영어 명사구/형용사구 (원문자 없이). "
        "**필수**: 본문에 ______ 가 정확히 한 번 등장해야 함. "
        "**금지**: 빈칸 ______ 외의 _..._ 밑줄 토큰, ①②③④⑤ 마커는 본문에 절대 넣지 말 것 "
        "(어법/어휘 패턴 혼용 금지)."
    ),
    "빈칸-절(32)": (
        "본문 한 곳에 빈칸 '______' (언더스코어 정확히 6개) 표시. 빈칸에 들어갈 절. "
        "question_text='다음 빈칸에 들어갈 말로 가장 적절한 것은?' "
        "선택지 5개는 영어 절 (원문자 없이). "
        "**중요**: 본문에 'that ______' 처럼 연결어가 빈칸 직전에 있다면 "
        "선택지에는 그 연결어를 **포함하지 말 것** (즉, choices 는 'that' 없이 절 본문만). "
        "**금지**: 빈칸 ______ 외의 _..._ 밑줄 토큰, ①②③④⑤ 마커는 본문에 절대 넣지 말 것."
    ),
    "빈칸-절(33)": (
        "빈칸-절(32)와 동일 형식, 다른 주제. choices 의 연결어 중복 금지 규칙도 동일."
    ),
    "빈칸-절(34)": (
        "빈칸-절(32)와 동일 형식, 다른 주제. choices 의 연결어 중복 금지 규칙도 동일."
    ),

    # ─── 논리 ────────────────────────────────────────────────────────────────
    "무관문장(35)": (
        "본문 5문장 중 흐름과 무관한 1개. "
        "question_text='다음 글에서 전체 흐름과 관계 없는 문장은?' "
        "**필수**: passage 구조는 [도입 문장 (번호 없음), '① 문장1', '② 문장2', '③ 무관 문장', '④ 문장4', '⑤ 문장5']. "
        "총 6원소(도입 + 번호 5개)이며 ①~⑤가 모두 본문에 등장해야 함. "
        "choices 는 정확히 ['①','②','③','④','⑤']. answer 는 무관 문장의 번호."
    ),
    "순서배열(36)": (
        "주어진 글 다음에 (A)(B)(C) 단락 순서를 결정. "
        "question_text='주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?' "
        "passage 는 **주어진 글의 본문만** (한 단락, 약 200~300자). 시스템이 박스로 감싸 렌더. "
        "sub_passages 는 [(A)단락 본문, (B)단락 본문, (C)단락 본문] 각 약 200~300자. "
        "**중요: sub_passages 안에 (A)/(B)/(C) 라벨을 넣지 말고 본문만 넣을 것** (라벨은 시스템이 자동 부착). "
        "choices 는 정확히 ['(A)-(C)-(B)','(B)-(A)-(C)','(B)-(C)-(A)','(C)-(A)-(B)','(C)-(B)-(A)']. "
        "**필수**: 각 보기는 반드시 괄호 포함 형식 '(X)-(Y)-(Z)' (예: '(B)-(A)-(C)'). "
        "괄호 없는 'B-A-C' 형식 절대 금지. "
        "group_label 은 null. 시스템이 동일 유형 연속 출제 시 자동 부착."
    ),
    "순서배열(37)": "순서배열(36)과 동일 형식 (다른 주제).",
    "문장삽입(38)": (
        "주어진 문장이 들어갈 위치 결정. "
        "question_text='글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳은?' "
        "given_sentence 는 삽입할 영어 문장 1개 (시스템이 박스로 감싸 렌더). "
        "**필수**: passage 안에 ①②③④⑤ 5개의 삽입 위치 마커가 모두 등장해야 함. "
        "예: '도입 문장. ① 첫째 문장. ② 둘째 문장. ③ 셋째 문장. ④ 넷째 문장. ⑤ 다섯째 문장.' "
        "choices 는 정확히 ['①','②','③','④','⑤']. "
        "group_label 은 null. 시스템이 자동 부착."
    ),
    "문장삽입(39)": "문장삽입(38)과 동일 형식 (다른 주제).",
    "요약문(40)": (
        "긴 본문 + 요약문 박스 (빈칸 (A), (B)) 완성. "
        "question_text='다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?' "
        "passage 는 본문. "
        "summary 는 'The text shows that (A) ______ leads to (B) ______ outcomes.' 형식 "
        "(빈칸은 정확히 언더스코어 6개씩 2곳). 시스템이 박스로 감싸 렌더. "
        "**필수**: summary 안에 ______ 가 정확히 2번, '(A)' '(B)' 라벨이 각 1번씩 등장. "
        "choices 5개는 (A)-(B) 단어 조합. 형식: '(A) word1 …… (B) word2'."
    ),
    "장문(41-42)": (
        "긴 본문 1개 + 2문항 세트 (41 제목 + 42 어휘). "
        "passage 는 본문 전체 (제목 + 본문).\n"
        "\n"
        "[(a)~(e) 어휘 후보 — 본문에 라벨만 박기, 마커는 시스템 담당]\n"
        "  본문 영어 문장 안에 라벨 5개를 단어 직전에 박는다 — '(a) ', '(b) ', ... 평문 형태.\n"
        "  예: 'Tradition often (a) confines creative expression. Scholars (b) embrace this view. "
        "Yet rigid rules can (c) stifle innovation. True artists (d) transcend boundaries, "
        "finding their voice through (e) experimentation.'\n"
        "  → '(a) confines' 같은 평문으로만. 시스템이 자동으로 '(a) _confines_' 처럼 underline 변환.\n"
        "  **금지**: 본문에 직접 _ 또는 *, 다른 마크다운 마커 사용 금지.\n"
        "  라벨 등장 순서: 본문 위→아래 (a)→(b)→(c)→(d)→(e) 알파벳 순.\n"
        "\n"
        "[출력 구조]\n"
        "  question_text='윗글의 제목으로 가장 적절한 것은?', choices=5개 영어 제목, answer=정수. "
        "  sub_questions=[{질문:'윗글의 밑줄 친 (a)~(e) 중에서 문맥상 낱말의 쓰임이 적절하지 않은 것은?', "
        "    choices:['(a)','(b)','(c)','(d)','(e)'], answer:정수}]. "
        "  group_label 은 null."
    ),
    "장문독해(43-45)": (
        "(A)(B)(C)(D) 4단락 본문 + 3문항 세트 (43 순서 + 44 지칭 + 45 일치). "
        "passage = (A) 단락 본문, sub_passages = [(B), (C), (D)]. 4단락 모두 비슷한 분량으로 균형있게. "
        "passage / sub_passages 안에 (A)~(D) 라벨 넣지 말 것 (시스템이 자동 부착).\n"
        "\n"
        "[(a)~(e) 지칭 — 본문에 라벨만 박기, 마커는 시스템 담당]\n"
        "  본문 영어 문장 안에 라벨 5개를 자연스럽게 박는다 — 단어 직전에 '(a) ', '(b) ', ... 형태.\n"
        "  예: 'Leo poured tea for (a) himself, lost in thought. Mr. Davies watched quietly. "
        "(b) He asked gently. Leo paused, weighing (c) his options. (d) He rarely doubted his path. "
        "But that day, (e) his certainty wavered.'\n"
        "  → '(a) himself' 같은 평문 형태로만. 시스템이 자동으로 '(a) _himself_' 처럼 underline 변환.\n"
        "  **금지**: 본문에 직접 _ 또는 *, 다른 마크다운 마커 사용 금지 — 평문 라벨만.\n"
        "  **라벨 등장 순서**: 본문 위→아래 (a)→(b)→(c)→(d)→(e) 알파벳 순.\n"
        "  passage(A)+sub_passages[0](B) 에 (a),(b) → sub_passages[1](C) 에 (c),(d) → "
        "  sub_passages[2](D) 에 (e). 정답 순서와 무관하게 단락 표기 순서대로.\n"
        "\n"
        "[44번 정답 구조 — 4:1]\n"
        "  5개 지칭 중 4개는 주인공 X, 1개는 부수 인물 Y. "
        "  Y 등장 라벨 위치 + 1 = sub_questions[0].answer. "
        "  예: assignments=['Leo','Mr.Harrison','Leo','Leo','Leo'], answer=2.\n"
        "  referent_assignments 5개 인물명을 정확히 명시 (validator 가 자동 4:1 검증, "
        "  거짓말 즉시 reject). 5:0 / 3:2 분포 금지.\n"
        "  **Y 라벨 직후 단어 규칙 (반드시)**: 부수 인물 Y 가 등장하는 그 1개 라벨 위치에는 "
        "  **대명사 (he/she/him/her/his/hers) 사용 금지**. 반드시 Y 의 이름/직함/명시적 호칭을 "
        "  라벨 직후에 박을 것.\n"
        "    ✓ 올바른 예: '(b) Mr. Davies watched the canvas thoughtfully.'\n"
        "    ✓ 올바른 예: '(b) the gallery owner stepped closer.'\n"
        "    ✓ 올바른 예: '(b) Professor Petrova nodded slowly.'\n"
        "    ❌ 잘못된 예: '(b) his mentor, Mr. Peterson, paid a visit.'  ← 'his' 는 X(Elias) 지칭\n"
        "    ❌ 잘못된 예: '(b) he watched quietly.'                      ← 누구 지칭인지 모호\n"
        "  X 가 등장하는 4개 라벨은 대명사(he/she/his/her) 자유 허용 — 위 규칙은 Y 1개만.\n"
        "  이유: 모델이 'X advised (b) him' 처럼 박으면 him 은 청자(=X) 가 되어 5:0 결함 발생. "
        "  Y 자체를 동작 주체/명시 호칭으로 박아야 모호성 제거.\n"
        "\n"
        "[지칭 변별력]\n"
        "  같은 성별 동성 2명으로 X, Y 를 잡아 단순 성별 비교로 안 풀리게. "
        "  'X advised (b) him' 에서 him 은 청자(주인공) — Y 지칭하게 하려면 "
        "  'X watched as (b) he frowned' 처럼 Y 자체를 동작 주체로.\n"
        "\n"
        "[출력 구조]\n"
        "  question_text='주어진 글 (A) 다음에 이어질 글의 순서로 가장 적절한 것은?', "
        "  choices=['(B)-(D)-(C)','(C)-(B)-(D)','(C)-(D)-(B)','(D)-(B)-(C)','(D)-(C)-(B)'], answer=정수. "
        "  sub_questions=[{질문:'밑줄 친 (a)~(e) 중에서 가리키는 대상이 나머지 넷과 다른 것은?', "
        "    choices:['(a)','(b)','(c)','(d)','(e)'], answer:정수}, "
        "  {질문:'윗글에 관한 내용으로 적절하지 않은 것은?', choices:5개 한국어 진술, answer:정수}]. "
        "  group_label 은 null."
    ),
}


@lru_cache(maxsize=1)
def _load_type_profiles() -> dict:
    """parser가 만든 templates/type_profiles.json 로드 (없으면 빈 dict)."""
    path = Path(__file__).resolve().parents[2] / "templates" / "type_profiles.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# 영어 평균 단어 길이 (공백 포함). 한국 수능 영어 지문 코퍼스에서 측정한 근사치.
# 글자수 통계 → 단어수 변환에만 사용 (모델에 글자수 노출하지 않기 위함).
_CHARS_PER_WORD = 6.0


def length_directive(q_type: str) -> str:
    """모델에 보여줄 길이 가이드. **글자수 통계는 노출하지 않음.**

    핵심 원리 (Phase 1):
      - 평균/허용범위/권장/soft_stop 4개 수치를 한꺼번에 보여주면 가장 큰 값에 anchor 가 잡힘.
      - 단어 수 1개 신호만 노출. 그것도 평균의 85% (overshoot bias 보정).
      - target_word_count 는 hard reject 기준이 아님 — 모델 자기 인식용 보조 신호.
      - 글자수 hard limit 은 validators 가 내부적으로 검증.
      - 길이는 좋은 scope 의 부산물이라는 메시지를 매번 박아 넣어 anchor 효과 차단.
    """
    prof = _load_type_profiles().get(q_type)
    if not prof:
        return ""

    src = prof.get("group_passage_len") or prof.get("passage_len") or {}
    avg = src.get("avg", 0)
    if avg <= 0:
        return ""

    # 글자수 평균 → 단어수 변환 후 0.85 ceiling (overshoot 편향 보정).
    target_words = max(50, int(avg / _CHARS_PER_WORD * 0.85))

    return (
        f"\n[글자수/단어수 가이드 — 강제 아닌 참고용]\n"
        f"  · target_word_count 는 약 {target_words} 단어 부근에서 자연스럽게 끝나는 값으로 정할 것.\n"
        f"  · 이 길이로 끝낼 수 없다면 plan.topic_scope 가 너무 넓다는 뜻이므로 scope 를 다시 좁힐 것.\n"
        f"\n"
        f"[절대 금지]\n"
        f"  · 분량을 채우기 위한 반복·곁가지 추가 (즉시 reject)\n"
        f"  · 분량을 줄이기 위한 논리 비약·결론 점프 (즉시 reject)\n"
        f"  · target_word_count 에 정확히 맞추려는 강박 (LLM 은 정확한 카운트 불가)\n"
        f"\n"
        f"길이는 좋은 scope 의 부산물이지 목표가 아님."
    )


# 하위 호환: 기존 호출부가 _length_hint 를 직접 부르고 있으면 새 함수로 redirect.
_length_hint = length_directive


def _recent_answers_directive(q_type: str) -> str:
    """순서배열(36,37) / 장문독해(43-45) 정답 분포 편향 회피 (Hotfix 17-2).

    측정 결과 장문독해 정답이 ①(B-D-C) 편향 — 13건 중 7건(54%, 평가원 기대 20%×2.7).
    Hotfix 7-3 의 강제 명령 버전은 9/9 실패라 폐기. 이번엔 약한 신호:
      · 최근 5건 success 의 sub_questions[0].answer 만 user 에 노출
      · '최근 정답이 X 였으니 이번엔 다른 번호로' 단순 명령
      · 본문 흐름 강제는 안 함 (인지 부담 회피)
    """
    if q_type not in ("순서배열(36)", "순서배열(37)", "장문독해(43-45)"):
        return ""
    import os
    log_path = os.environ.get("USAGE_LOG_PATH")
    if log_path:
        path = Path(log_path)
    else:
        path = Path(__file__).resolve().parents[2] / "tools" / "stats" / "usage_log.jsonl"
    if not path.exists():
        return ""
    try:
        recent: list[int] = []
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != q_type or not rec.get("success"):
                continue
            ans = rec.get("answer")
            if isinstance(ans, int) and 1 <= ans <= 5:
                recent.append(ans)
            if len(recent) >= 5:
                break
        if not recent:
            return ""
    except Exception:
        return ""

    # 정답 라벨로 변환
    if q_type == "장문독해(43-45)":
        labels = ['(B)-(D)-(C)','(C)-(B)-(D)','(C)-(D)-(B)','(D)-(B)-(C)','(D)-(C)-(B)']
    else:
        labels = ['(A)-(C)-(B)','(B)-(A)-(C)','(B)-(C)-(A)','(C)-(A)-(B)','(C)-(B)-(A)']
    listed = ", ".join(f"①②③④⑤"[a-1] + labels[a-1] for a in recent)
    return (
        f"\n\n[최근 정답 회피 — 약한 신호]\n"
        f"  최근 success 5건 정답: {listed}\n"
        f"  같은 번호로만 쏠리지 않게 위와 다른 정답 번호로 출제 권장 (강제 아님)."
    )


def _ref_context(reference_text: str) -> str:
    if not reference_text:
        return ""
    return f"\n\n[참고 자료 - 이 내용을 바탕으로 관련 주제/어휘를 활용하세요]\n{reference_text[:2000]}"


# 최근 출제된 scope 회피 — 동일 유형에서 narrative 수렴(예: 장문독해의 'young artist'
# modal pattern) 을 자동으로 깨기 위한 메커니즘.
_RECENT_SCOPE_LIMIT = 5  # 최근 N건 scope 를 user 메시지에 노출


def _answer_diversity_directive(q_type: str) -> str:
    """순서배열(36,37) / 장문독해(43-45) 의 정답 다양성 강제 — 강력 명령형 버전.

    Hotfix 8 에서 비활성됐던 함수. 강제 명령("자연스럽게 이어지도록")이 인지 부담 폭증
    시켜 9/9 실패. 현재 비활성 상태 — prompt_for_type 에서 호출 안 함. 보존만.
    """
    import random
    if q_type in ("순서배열(36)", "순서배열(37)"):
        # 보기: ①(A)-(C)-(B), ②(B)-(A)-(C), ③(B)-(C)-(A), ④(C)-(A)-(B), ⑤(C)-(B)-(A)
        # ①은 5개 옵션 중 하나일 뿐 — 5개 균등 분포로 무작위 선택
        target = random.randint(1, 5)
        circled = "①②③④⑤"[target - 1]
        choices = [
            "(A)-(C)-(B)", "(B)-(A)-(C)", "(B)-(C)-(A)",
            "(C)-(A)-(B)", "(C)-(B)-(A)",
        ]
        return (
            f"\n\n[이번 호출 정답 분포 강제]\n"
            f"  · 이 문제의 정답은 **{circled} {choices[target-1]}** 으로 만들 것 (answer={target}).\n"
            f"  · sub_passages 는 (A)/(B)/(C) 순서로 입력되지만, 자연스러운 흐름은 "
            f"위 정답 순서대로 읽어야 성립해야 한다.\n"
            f"  · 즉 sub_passages 자체는 (A)→(B)→(C) 인데 학생이 정답을 따라 읽으면 "
            f"{choices[target-1]} 흐름이 자연스럽게 이어지도록 작성.\n"
            f"  · 모델 편향상 항상 ①(B-C-D) 으로 수렴하는 경향이 있어 이번 호출은 "
            f"강제로 다른 정답 분포로 출제."
        )
    if q_type == "장문독해(43-45)":
        # 보기: ①(B)-(D)-(C), ②(C)-(B)-(D), ③(C)-(D)-(B), ④(D)-(B)-(C), ⑤(D)-(C)-(B)
        target = random.randint(1, 5)
        circled = "①②③④⑤"[target - 1]
        choices = [
            "(B)-(D)-(C)", "(C)-(B)-(D)", "(C)-(D)-(B)",
            "(D)-(B)-(C)", "(D)-(C)-(B)",
        ]
        return (
            f"\n\n[이번 호출 정답 분포 강제]\n"
            f"  · 43번 (순서) 의 정답은 **{circled} {choices[target-1]}** 으로 만들 것 (answer={target}).\n"
            f"  · sub_passages 는 [(B)본문, (C)본문, (D)본문] 순서로 입력되지만, "
            f"학생이 (A) 다음에 정답대로 읽으면 자연스러운 흐름이 되도록 본문을 설계.\n"
            f"  · sub_passages 자체는 (B)/(C)/(D) 순서지만 그 안의 내용은 "
            f"(A)→{choices[target-1]} 흐름이 자연스럽도록 분배.\n"
            f"  · 모델 편향(항상 ①) 을 이번 호출은 강제로 다른 분포로 출제."
        )
    return ""


def _recent_scopes_directive(q_type: str) -> str:
    """usage_log.jsonl 에서 동일 유형 최근 성공 호출의 plan_topic_scope 를 읽어
    user 메시지에 회피 가이드로 첨부.

    설계:
      - 같은 type 에서 success=True 인 호출만 (실패한 scope 는 회피 대상 아님).
      - 최신 N건 (역순으로 읽되 중복 scope 는 dedup).
      - 로그 파일 없거나 읽기 실패 → 빈 문자열 반환 (안전).
      - plan 필드가 없는 baseline 행은 자동 무시 (plan_topic_scope 키 없음).
    """
    import os
    log_path = os.environ.get("USAGE_LOG_PATH")
    if log_path:
        path = Path(log_path)
    else:
        path = Path(__file__).resolve().parents[2] / "tools" / "stats" / "usage_log.jsonl"
    if not path.exists():
        return ""
    try:
        scopes: list[str] = []
        seen: set[str] = set()
        # 파일 끝에서부터 읽기 (최신 우선)
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != q_type or not rec.get("success"):
                continue
            scope = rec.get("plan_topic_scope")
            if not scope or scope in seen:
                continue
            seen.add(scope)
            scopes.append(scope)
            if len(scopes) >= _RECENT_SCOPE_LIMIT:
                break
        if not scopes:
            return ""
    except Exception:
        return ""

    # 길이 한도 — 너무 길면 user 메시지가 부풀어 attention 분산
    listed = "\n".join(f"  · {s[:200]}" for s in scopes)
    return (
        f"\n\n[최근 출제된 주제 — 이와 다른 narrative 로 작성할 것]\n"
        f"{listed}\n"
        f"위 주제·등장인물·setting 과 겹치지 않는 새로운 scope 로 plan.topic_scope 를 잡을 것. "
        f"같은 직업군(예: 예술가)·같은 plot 구조(예: 슬럼프→멘토→깨달음) 반복 금지."
    )


def _base_system() -> str:
    return SYSTEM_PROMPT


def is_supported(q_type: str) -> bool:
    return q_type in TYPE_HINTS


def prompt_for_type(
    q_type: str, grade: str, points: float, reference_text: str = "",
) -> tuple[str, str]:
    """(system, user) 메시지 반환. 출력 스키마는 호출자가 provider에 직접 강제."""
    if q_type in DISABLED_TYPES:
        raise NotImplementedError(
            f"유형 '{q_type}'는 현재 미지원입니다 (이미지 생성 후속 작업 필요)."
        )
    if q_type not in TYPE_HINTS:
        raise ValueError(
            f"알 수 없는 유형: {q_type!r}. 지원 유형: {ACTIVE_TYPES}"
        )

    grade_hint = GRADE_HINT.get(grade, GRADE_HINT["고3 (수능)"])
    type_hint  = TYPE_HINTS[q_type]
    length     = _length_hint(q_type)
    ref        = _ref_context(reference_text)
    avoid      = _recent_scopes_directive(q_type)       # 동일 유형 최근 scope 회피
    ans_avoid  = _recent_answers_directive(q_type)      # 정답 편향 회피 (Hotfix 17-2)

    # _answer_diversity_directive (Hotfix 7-3) 는 영구 비활성. 본문 흐름 강제 명령이
    # 인지 부담 폭증으로 9/9 실패. 이번엔 _recent_answers_directive 만 사용 (약한 신호).

    user = (
        f"수능 영어 '{q_type}' 유형 문제를 1개 생성해주세요.\n"
        f"난이도: {grade_hint}\n"
        f"배점: {points}점\n\n"
        f"유형 가이드: {type_hint}{length}{avoid}{ans_avoid}{ref}"
    )
    return SYSTEM_PROMPT, user
