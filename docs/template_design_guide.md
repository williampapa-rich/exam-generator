# Template 디자인 수정 가이드

`templates/template.hwpx`의 디자인을 수정하려면 **한글에서 직접 수정 → set-template으로 갱신**하는 흐름을 따릅니다.

> ⚠️ template.hwpx 는 SSOT 입니다. parser CLI 의 `set-template` 명령 외에는 절대 덮어쓰지 마세요.
> renderer 는 이 파일의 placeholder만 치환할 뿐, 디자인을 만지지 않습니다.

---

## 작업 흐름

1. 원본 보호용 사본 만들기 (선택):
   ```bash
   cp templates/template.hwpx ~/Desktop/template_backup_$(date +%Y%m%d).hwpx
   ```
2. 한글로 `templates/template.hwpx` 열기.
3. 아래 D1/D2/D3 항목 수정 후 저장.
4. 수정한 hwpx로 SSOT 갱신:
   ```bash
   python -m tools.parser.cli set-template templates/template.hwpx
   ```
   (또는 임시 위치에 저장한 hwpx 경로 지정)
5. uvicorn 서버는 `--reload` 가 hwpx 변경을 감지하지 못하므로 **수동 재시작** 필요.
   ```bash
   pkill -f "uvicorn app.main:app"
   .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
6. 시험지 생성 후 한글에서 결과 확인.

---

## D1. 2페이지 머릿글 ↔ 본문 사이 가로 border 위치 보정

**증상**: 2페이지(짝수 페이지)의 머릿글 영역 하단과 본문 시작 사이에 그려지는 가로선이 머릿글 영역 하단 라인과 어긋나 있음.

**작업 위치**: 2페이지의 머릿글/본문 경계.
- 한글에서 [보기] → [편집 화면 보기] 또는 [페이지 윤곽] 켜기.
- 짝수 페이지를 표시하기 위해 페이지 추가가 필요할 수 있음 (template은 1페이지 기본).

**수정 방법**: 한글 [쪽] → [편집용지] → [머리말/꼬리말] 또는 가로선이 그려진 도형/표 테두리의 좌표를 머릿글 하단 baseline 과 일치시킴.

**검증**: 시험지 생성 후 2페이지 이상 분량으로 출력해서 가로선과 머릿글 라벨 하단이 일치하는지 확인.

---

## D2. 머릿글 좌상단/우상단 라벨의 bottom margin 보정

**증상**: `{{TOP_LEFT_LABEL}}`, `{{TOP_RIGHT_LABEL}}` 텍스트의 bottom margin이 머릿글 가로 border 라인과 떨어져 있음.

**작업 위치**: 1페이지 + 2페이지 모두 (masterpage0, masterpage1, masterpage2 에 동일하게 적용됨).

**수정 방법**:
1. 한글에서 머릿글 영역 더블클릭으로 편집 모드 진입.
2. 좌상단/우상단 라벨 텍스트가 들어 있는 단락의 [단락 모양] → [여백/간격] → [문단 아래 간격] 을 0 또는 최소값으로 설정.
3. 또는 라벨이 표 안에 있다면 셀의 [셀 모양] → [여백/간격] → 아래 여백을 0 으로 설정.

**검증**: 머릿글 라벨 텍스트의 baseline 과 가로 border 라인이 거의 붙어있어야 함.

---

## D3. 마지막 페이지 우측 단 하단 "확인 사항" 박스 — 잠금 해제 + 위치 보정

**증상**:
1. 박스가 워터마크처럼 잠겨서 사용자가 한글에서 텍스트를 클릭/수정할 수 없음.
2. 위치가 약 2px 아래에 있어 위로 올려야 함.

**작업 위치**: 마지막 페이지 (현재 template은 1페이지짜리지만 장문 시험지에서는 마지막 페이지)의 우측 단 하단.

### D3-1. 잠금 해제

1. 한글에서 박스 객체를 우클릭 → [개체 속성] → [기본] 탭.
2. **[잠금]** 또는 **[보호]** 체크 해제.
3. 박스가 워터마크 처리되어 있다면 [도구] → [고급 → 보호] 에서 워터마크 해제.

### D3-2. 위치 보정

1. 박스를 클릭으로 선택.
2. [개체 속성] → [기본] 탭에서 위치(Y 좌표)를 약 2px(=약 56 HWP unit) 위로 조정.
   - 또는 박스를 직접 드래그해서 시각적으로 맞추기.

**검증**:
- 한글에서 박스 안 텍스트를 클릭했을 때 커서가 들어가서 글자 수정 가능.
- 박스가 미세하게 위로 올라간 위치에 있음.

---

## 변경 후 SSOT 갱신 명령 (재인용)

```bash
cd ~/workspace/exam-generator
.venv/bin/python -m tools.parser.cli set-template templates/template.hwpx
```

이 명령은 다음 작업을 수행합니다:
- `templates/template.hwpx` 자체는 그대로 (이미 사용자가 수정한 파일이므로)
- `templates/header.xml` 을 새로 추출 (템플릿 안의 header.xml 기반)
- `tools/parser/_data/samples_log.json` 에 set-template 이력 기록

---

## 주의사항

- placeholder (`{{Qn_GROUP_LABEL}}`, `{{SUBJECT_NAME}}` 등) 의 **이름/위치/철자를 절대 바꾸지 마세요**.
  injector 가 정확히 이 패턴을 찾아 치환합니다.
- 단락 갯수도 보존:
  - 머릿글 영역: 8단락 (idx 0~7)
  - Q1~Q10 슬롯: 각 10단락 = 100단락 (idx 8~107)
  - 마지막 trailing: 1단락 (idx 108)
- 슬롯 수 (현재 10개) 를 늘리거나 줄이려면 코드 수정 필요 → 작업 전 알리기.
