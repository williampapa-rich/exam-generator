"""
시험지 생성 FastAPI 서버
실행: uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from dotenv import load_dotenv

from app.renderer.builder import build_hwpx
from app.llm.base         import get_client
from app.llm.generator    import generate_all
from app.llm.pdf          import extract_reference_text
from shared.schemas.question import GenerateRequest

load_dotenv()

app = FastAPI(title="수능 영어 시험지 생성기", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = Path(__file__).parent / "web" / "index.html"
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.post("/api/upload-ref")
async def upload_reference(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일 크기가 50MB를 초과합니다.")

    text = await extract_reference_text(pdf_bytes)
    return JSONResponse({"reference_text": text, "filename": file.filename})


@app.post("/api/generate")
async def generate_exam(req: GenerateRequest):
    if not req.questions:
        raise HTTPException(status_code=400, detail="문제를 1개 이상 추가해주세요.")

    try:
        client = get_client(req.provider, req.model) if (req.provider or req.model) else None
        questions = await generate_all(
            req.questions, req.meta.grade, req.reference_text, client=client,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 생성 실패: {e}")

    hwpx_buf = build_hwpx(req.meta, questions)

    safe_title = req.meta.title.replace(" ", "_").replace("/", "-")
    filename   = f"{safe_title}_{req.meta.date}.hwpx"

    return StreamingResponse(
        hwpx_buf,
        media_type="application/x-hwp+zip",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "X-Question-Count": str(len(questions)),
        },
    )


@app.get("/api/health")
async def health():
    api_key_ok = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "status":    "ok",
        "api_key":   "설정됨" if api_key_ok else "⚠️ .env 파일에 GEMINI_API_KEY 필요",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
