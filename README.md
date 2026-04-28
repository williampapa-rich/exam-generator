# exam-generator

수능 영어 시험지 자동 생성기 (FastAPI + Multi-LLM).

학원 영어 강사용 HWPX 시험지 생성 도구. Parser가 추출한 표준 템플릿/스키마에 LLM이 생성한 문제를 렌더링합니다.

## 구조

- `app/` — FastAPI 웹앱 (배포 대상)
  - `renderer/` — HWPX ZIP 빌더
  - `llm/` — Multi-LLM 어댑터 (Gemini / OpenAI / Anthropic)
- `shared/schemas/` — Pydantic SSOT
- `templates/` — Parser 산출물 (template.hwpx, schema.json 등)
- `tools/parser/` — 어드민 도구 (HWPX 분석 + 산출물 생성, 비배포)
- `samples/` — 학습용 HWPX 원본

## 설치

```bash
# 배포 런타임만
pip install -e .

# 어드민 + 개발 환경
pip install -e ".[parser,dev]"
```

## 실행

```bash
cp .env.example .env
# .env 열어서 LLM_PROVIDER + 해당 API_KEY 입력

uvicorn app.main:app --reload --port 8000
```

브라우저 → http://localhost:8000

## Parser 사용 (템플릿 갱신 시)

```bash
python -m tools.parser.cli build samples/평가원_영어_양식.hwpx --out templates/
```
