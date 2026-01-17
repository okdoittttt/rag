# Terminal RAG

터미널에서 동작하는 RAG (Retrieval-Augmented Generation) 시스템.  
로컬 문서를 인덱싱하고 자연어 질의로 근거 기반 답변을 생성합니다.

---

## 🎯 프로젝트 목표

- **재현 가능**: 동일 입력 → 동일 결과
- **안전**: 민감정보 보호, 최소 권한 실행
- **확장 가능**: 다양한 문서 형식 지원
- **평가 가능**: 품질 측정 및 회귀 방지

---

## 📋 구현 로드맵

### Phase 1: 기반 구축
- [x] 프로젝트 구조 설계 (`src/`, `cli/`, `configs/`, `data/`, `tests/`)
- [x] 설정 파일 구조 정의 (`configs/default.yaml`)
- [x] 로깅 시스템 구축 (구조화 JSON 로그)

### Phase 2: 문서 수집 (Ingestion)
- [x] 텍스트 파일 로더 구현 (`.txt`, `.md`)
- [x] 문서 정규화 (공백/개행 정리, 메타데이터 추출)
- [x] 언어 감지 (한국어/영어)

### Phase 3: 청킹 (Chunking)
- [x] 기본 청킹 로직 (300-800 토큰 또는 800-2000자)
- [x] 오버랩 처리 (10-20%)
- [x] 구조 보존 (헤더/섹션 경계)T

### Phase 4: 임베딩 & 인덱싱
- [x] 임베딩 모델 선정 및 연동
- [x] 벡터 저장소 구현 (로컬 기반: FAISS 또는 Chroma)
- [x] 인덱스 저장/로드 기능

### Phase 5: 검색 (Retrieval)
- [ ] 벡터 유사도 검색 구현
- [ ] Top-K, score threshold 설정
- [ ] (선택) BM25 키워드 검색 → 하이브리드

### Phase 6: 답변 생성 (Generation)
- [ ] LLM 연동 (OpenAI API 또는 로컬 모델)
- [ ] 프롬프트 템플릿 설계 (근거 인용 강제)
- [ ] 컨텍스트 길이 초과 시 요약/압축

### Phase 7: CLI 인터페이스
- [ ] 문서 인덱싱 명령어 (`rag index <path>`)
- [ ] 질의 명령어 (`rag ask "<question>"`)
- [ ] 상태 확인 명령어 (`rag status`)

### Phase 8: 평가 (Evaluation)
- [ ] 평가 데이터셋 구축 (질문-정답 쌍)
- [ ] 평가 지표 구현 (정답률, 인용 정확도, 응답 지연)
- [ ] 회귀 테스트 자동화

---

## 🛠 기술 스택

| 구분 | 도구 |
|------|------|
| 언어 | Python 3.12 |
| 패키지 관리 | uv |
| CLI | typer (예정) |
| 벡터 저장소 | FAISS 또는 Chroma (예정) |
| 임베딩 | OpenAI / sentence-transformers (예정) |
| LLM | OpenAI API 또는 Ollama (예정) |

---

## 🚀 시작하기

```bash
# 의존성 설치
uv sync

# 가상환경 활성화
source .venv/bin/activate

# (구현 후) 문서 인덱싱
rag index ./docs

# (구현 후) 질의
rag ask "이 프로젝트의 목표는?"
```

---

## 📁 프로젝트 구조 (예정)

```
rag/
├── src/              # 핵심 라이브러리
│   ├── ingestion/    # 문서 수집
│   ├── chunking/     # 청킹
│   ├── embedding/    # 임베딩
│   ├── retrieval/    # 검색
│   └── generation/   # 답변 생성
├── cli/              # CLI 엔트리포인트
├── configs/          # 설정 파일
├── data/             # 데이터 (인덱스, 캐시)
├── eval/             # 평가 스크립트
├── tests/            # 테스트
└── docs/             # 문서
```

---

## 📝 License

MIT
