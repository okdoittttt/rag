/**
 * API 호출 함수 모듈
 */

export interface AskOptions {
    topK?: number;
    rerank?: boolean;
    expand?: boolean;
    provider?: "gemini" | "ollama";
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
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ message: "Unknown error" }));
        throw new Error(error.message || `API error: ${response.status}`);
    }

    return response.json();
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
