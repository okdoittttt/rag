/**
 * API 호출 함수 모듈
 */

export interface AskOptions {
    topK?: number;
    rerank?: boolean;
    expand?: boolean;
    provider?: "gemini" | "ollama";
    user_id?: string;  // 사용자 ID (격리된 검색용)
    source_filter?: string;  // 특정 문서로 검색 제한 (파일명)
    doc_mode?: boolean;  // 문서 모드 (요약 의도 시 전체 청크 투입)
    summarize_override?: boolean | null;  // 요약 모드 강제 토글 (null=휴리스틱 위임)
    api_key?: string;
    model_name?: string;
    base_url?: string;
}

export interface ChunkReference {
    content: string;
    source: string;
    score: number;
}

export interface AskResponse {
    answer: string;
    references: ChunkReference[];
}

export interface SearchResponse {
    results: ChunkReference[];
}

/**
 * RAG 질문-답변 API 호출
 */
export async function askQuestion(
    query: string,
    options: AskOptions = {}
): Promise<AskResponse> {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            query,
            top_k: options.topK ?? 5,
            rerank: options.rerank ?? true,
            expand: options.expand ?? false,
            provider: options.provider ?? "gemini",
            user_id: options.user_id,
            source_filter: options.source_filter,
            doc_mode: options.doc_mode ?? false,
            summarize_override: options.summarize_override ?? null,
            api_key: options.api_key,
            model_name: options.model_name,
            base_url: options.base_url,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ message: "Unknown error" }));
        throw new Error(error.message || `API error: ${response.status}`);
    }

    return response.json();
}

/**
 * 스트리밍 질문 답변 API 호출
 *
 * 백엔드 `/ask/stream`(SSE)을 호출하여 진행 단계·추론·답변 이벤트를 콜백으로 전달한다.
 *
 * @param onChunk 최종 답변 토큰(`{text}`)이 도착할 때마다 호출.
 * @param onReferences 참조 청크 목록(`{references}`)이 도착할 때 호출.
 * @param onComplete `[DONE]` 수신 시 호출.
 * @param onError 오류 발생 시 호출.
 * @param onReasoning CoT 추론 토큰(`{phase:"reasoning", text}`) 도착 시 호출(선택).
 * @param onStatus 진행 단계(`searching`/`analyzing`/`reasoning_start` 등) 도착 시 호출(선택).
 */
export async function askQuestionStream(
    query: string,
    options: AskOptions,
    onChunk: (text: string) => void,
    onReferences: (refs: ChunkReference[]) => void,
    onComplete: () => void,
    onError: (err: Error) => void,
    onReasoning?: (text: string) => void,
    onStatus?: (phase: string) => void
): Promise<void> {
    try {
        const response = await fetch("/api/ask/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                query,
                top_k: options.topK ?? 5,
                rerank: options.rerank ?? true,
                expand: options.expand ?? false,
                provider: options.provider ?? "gemini",
                user_id: options.user_id,
                source_filter: options.source_filter,
                doc_mode: options.doc_mode ?? false,
                summarize_override: options.summarize_override ?? null,
                api_key: options.api_key,
                model_name: options.model_name,
                base_url: options.base_url,
            }),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: "Unknown error" }));
            throw new Error(error.message || `API error: ${response.status}`);
        }

        if (!response.body) {
            throw new Error("Response body is empty");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || ""; // Incomplete line

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const data = line.slice(6);
                    if (data === "[DONE]") {
                        onComplete();
                        return;
                    }

                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.phase === "error") {
                            onError(new Error(parsed.detail || "답변 생성 중 오류가 발생했습니다."));
                            return;
                        } else if (parsed.phase === "reasoning") {
                            // CoT 추론 토큰
                            if (parsed.text) onReasoning?.(parsed.text);
                        } else if (parsed.references) {
                            onReferences(parsed.references);
                        } else if (parsed.text) {
                            // 최종 답변 토큰 (phase 없음)
                            onChunk(parsed.text);
                        } else if (parsed.phase) {
                            // 진행 단계 (searching/analyzing/reasoning_start 등)
                            onStatus?.(parsed.phase);
                        }
                    } catch (e) {
                        console.error("Failed to parse SSE data:", data, e);
                    }
                }
            }
        }
    } catch (err) {
        onError(err instanceof Error ? err : new Error("Unknown error during streaming"));
    }
}

/**
 * 문서 검색 API 호출
 */
export async function searchDocuments(
    query: string,
    topK: number = 5
): Promise<SearchResponse> {
    const response = await fetch("/api/search", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            query,
            top_k: topK,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ message: "Unknown error" }));
        throw new Error(error.message || `API error: ${response.status}`);
    }

    return response.json();
}

export async function getSystemPrompt(): Promise<string> {
    const response = await fetch("/api/config/system-prompt");
    if (!response.ok) {
        throw new Error("Failed to fetch system prompt");
    }
    const data = await response.json();
    return data.system_prompt;
}

export async function updateSystemPrompt(prompt: string): Promise<void> {
    const response = await fetch("/api/config/system-prompt", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ system_prompt: prompt }),
    });
    if (!response.ok) {
        throw new Error("Failed to update system prompt");
    }
}
