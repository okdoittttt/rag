/**
 * API 호출 함수 모듈
 */

export interface AskOptions {
    topK?: number;
    rerank?: boolean;
    expand?: boolean;
    provider?: "gemini" | "ollama";
    user_id?: string;  // 사용자 ID (격리된 검색용)
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
            rerank: options.rerank ?? false,
            expand: options.expand ?? false,
            provider: options.provider ?? "gemini",
            user_id: options.user_id,
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
 */
export async function askQuestionStream(
    query: string,
    options: AskOptions,
    onChunk: (text: string) => void,
    onReferences: (refs: ChunkReference[]) => void,
    onComplete: () => void,
    onError: (err: Error) => void
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
                rerank: options.rerank ?? false,
                expand: options.expand ?? false,
                provider: options.provider ?? "gemini",
                user_id: options.user_id,
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
                        if (parsed.text) {
                            onChunk(parsed.text);
                        } else if (parsed.references) {
                            onReferences(parsed.references);
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
