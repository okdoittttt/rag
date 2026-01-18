# FastAPI REST API 구현 계획

RAG 시스템을 FastAPI 기반 REST API로 확장하는 단계별 구현 계획입니다.

---

## 📋 구현 단계

### Phase 1: 프로젝트 구조 설정 ✅
- [x] `src/api/` 디렉토리 생성
- [x] FastAPI 앱 초기화 (`src/api/main.py`)
- [x] 의존성 추가 (`fastapi`, `uvicorn`)
- [x] 기본 Health Check 엔드포인트 구현

### Phase 2: 핵심 엔드포인트 구현 ✅
- [x] **POST /ask** - 질문-답변 (RAG 파이프라인 전체 실행)
- [x] **POST /search** - 검색만 수행 (답변 생성 없이 관련 문서 반환)
- [x] **POST /index** - 문서 인덱싱 (파일 업로드 또는 텍스트 직접 전달)

### Phase 3: 스키마 및 설정 ✅
- [x] Pydantic Request/Response 스키마 정의 (`src/api/schemas.py`)
- [x] CORS 미들웨어 설정
- [x] 에러 핸들링 및 표준 응답 포맷 (`src/api/exceptions.py`)

### Phase 4: 스트리밍 응답 (선택)
- [ ] SSE(Server-Sent Events) 기반 스트리밍 답변
- [ ] `/ask/stream` 엔드포인트 추가

### Phase 5: 배포 준비
- [ ] Docker Compose에 API 서비스 추가
- [ ] 환경변수 기반 설정 분리
- [ ] Swagger 문서 확인 및 정리

---

## 🔌 API 엔드포인트 설계

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/ask` | 질문에 대한 답변 생성 |
| POST | `/search` | 관련 문서 검색만 |
| POST | `/index` | 문서 인덱싱 |
| POST | `/ask/stream` | 스트리밍 답변 (선택) |

---

## 📁 디렉토리 구조 (예정)

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI 앱 진입점
│   ├── routes/
│   │   ├── ask.py       # /ask 라우터
│   │   ├── search.py    # /search 라우터
│   │   └── index.py     # /index 라우터
│   ├── schemas.py       # Pydantic 모델
│   └── dependencies.py  # 의존성 주입
└── rag/                  # 기존 RAG 로직 (변경 없음)
```

---

## ⏱️ 예상 일정

| Phase | 예상 소요 |
|-------|----------|
| Phase 1 | 1시간 |
| Phase 2 | 2-3시간 |
| Phase 3 | 1시간 |
| Phase 4 | 2시간 (선택) |
| Phase 5 | 1시간 |

---
src/api/schemas.py - Pydantic 스키마
src/api/routes/ask.py - /ask 엔드포인트
src/api/routes/search.py - /search 엔드포인트
src/api/routes/index.py - /index 엔드포인트
*작성일: 2026-01-18*