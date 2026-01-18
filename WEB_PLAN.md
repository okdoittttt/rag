# Web UI 구현 계획

RAG 시스템을 위한 Next.js 기반 웹 인터페이스 구현 계획입니다.
`ui/` 디렉토리에 구축하며, TailwindCSS를 사용합니다.

---

## 📋 구현 단계

### Phase 1: 프로젝트 초기화 ✅
- [x] `ui/` 디렉토리에 Next.js App Router 프로젝트 생성
- [x] TailwindCSS, Lucide-React(아이콘) 설정
- [x] Proxy 설정 (`next.config.ts`) - API 서버(8000) 연동
- [x] 기본 레이아웃 (Header, Main Container)
- [x] 로컬 폰트(Paperlogy) 적용

### Phase 2: 검색 및 채팅 인터페이스 ✅
- [x] **ChatInput.tsx**: 질문 입력 컴포넌트
- [x] **ChatList.tsx**: 대화 목록 표시 영역
- [x] **ChatMessage.tsx**:
    - 질문 (User)
    - 답변 (Bot) - Markdown 렌더링 지원 (react-markdown)
    - 로딩 인디케이터
    - 참조 문서 표시

### Phase 3: API 연동 ✅
- [x] `lib/api.ts` 구현 (askQuestion, searchDocuments)
- [x] **POST /ask** 연동
- [x] **ModelSelector** 컴포넌트 (Gemini / Ollama 전환)
- [x] Docker Compose 환경변수 설정 (Qdrant, Ollama, Gemini)

### Phase 4: 스트리밍 답변 (SSE) ⏳
- [x] `askQuestionStream` 함수 구현 (api.ts)
- [x] Edge Route Proxy (`app/api/ask/stream/route.ts`)
- [ ] 실시간 텍스트 타이핑 효과 검증
- [ ] 실시간 참조 문서(Source) 표시 검증

### Phase 5: Docker 통합 및 배포
- [x] `ui/Dockerfile` 작성 (standalone 빌드)
- [ ] `compose.yaml`에 UI 서비스 추가
- [ ] API URL 환경변수화 (`API_BASE_URL`)

### Phase 6: 설정 및 사이드바 (New)
- [ ] **Sidebar**: 좌측 사이드바 (채팅 목록, 설정 진입)
- [ ] **Settings Modal/Page**:
    - Gemini API Key / Model 설정
    - Ollama URL / Model 설정
    - 로컬 스토리지에 저장하여 사용



---

## 📁 디렉토리 구조 (예정)

```
ui/
├── app/
│   ├── page.tsx          # 메인 채팅 화면
│   ├── layout.tsx        # 글로벌 레이아웃
│   └── globals.css       # Tailwind 설정
├── components/
│   ├── chat/             # 채팅 관련 컴포넌트
│   │   ├── ChatList.tsx
│   │   ├── ChatMessage.tsx
│   │   └── ChatInput.tsx
│   ├── common/           # 공통 컴포넌트 (Button, Input 등)
│   └── layout/           # Header, Sidebar
├── lib/
│   ├── api.ts            # API 호출 함수
│   └── hooks/            # 커스텀 훅 (useChat)
└── next.config.ts        # 프록시 설정
```

---

## ⏱️ 예상 일정

| Phase | 예상 소요 |
|-------|----------|
| Phase 1 | 30분 |
| Phase 2 | 1-2시간 |
| Phase 3 | 1시간 |
| Phase 4 | 2시간 |
| Phase 5 | 1시간 |

---

*작성일: 2026-01-18*
