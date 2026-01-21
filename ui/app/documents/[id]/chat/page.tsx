"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { ArrowLeft, FileText } from "lucide-react";
import Link from "next/link";
import { ChatList, ChatInput, Message } from "@/components/chat";
import { ModelSelector } from "@/components/chat/ModelSelector";
import { askQuestionStream } from "@/lib/api";
import { useSettingsStore } from "@/lib/store";
import { useChatStore } from "@/lib/chatStore";

export default function DocumentChatPage() {
    const router = useRouter();
    const params = useParams();
    const searchParams = useSearchParams();
    const { data: session } = useSession();
    const settings = useSettingsStore();
    const chatStore = useChatStore();

    const documentId = params.id as string;
    const filename = searchParams.get("filename") || "문서";

    // Store current session ID ref to prevent stale closures
    const sessionIdRef = useRef<string | null>(null);

    const [isLoading, setIsLoading] = useState(false);
    const [provider, setProvider] = useState<"gemini" | "ollama">("gemini");
    const [expand, setExpand] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        // Create a new session specifically for this document if one doesn't exist
        // or just use a new session. For simplicity, we'll start a new session.
        // Ideally, we could check if there's an existing session for this document.
        const newSessionId = chatStore.createSession();
        chatStore.updateSessionTitle(newSessionId, `${filename} (Document Chat)`);
        chatStore.setCurrentSession(newSessionId);
        sessionIdRef.current = newSessionId;

        return () => {
            // Optional: Cleanup empty sessions?
        };
    }, [filename]);

    const messages = mounted && chatStore.currentSessionId && chatStore.sessions[chatStore.currentSessionId]
        ? chatStore.sessions[chatStore.currentSessionId].messages
        : [];

    const handleSubmit = async (query: string) => {
        const sessionId = sessionIdRef.current || chatStore.currentSessionId;
        if (!sessionId) return; // Should not happen

        // Add user message
        const userMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: query,
        };
        chatStore.addMessage(sessionId, userMessage);

        // Add loading bot message
        const botId = (Date.now() + 1).toString();
        const botMessage: Message = {
            id: botId,
            role: "assistant",
            content: "",
            isLoading: true
        };
        chatStore.addMessage(sessionId, botMessage);

        setIsLoading(true);

        try {
            await askQuestionStream(
                query,
                {
                    provider,
                    expand,
                    user_id: session?.user?.id,
                    source_filter: filename,  // 특정 문서로 제한
                    api_key: provider === "gemini" ? settings.geminiApiKey : undefined,
                    model_name: provider === "gemini" ? settings.geminiModel : settings.ollamaModel,
                    base_url: provider === "ollama" ? settings.ollamaBaseUrl : undefined,
                },
                (text) => {
                    chatStore.appendMessageContent(sessionId, botId, text);
                },
                (refs) => {
                    chatStore.updateMessage(sessionId, botId, { references: refs });
                },
                () => {
                    setIsLoading(false);
                    chatStore.updateMessage(sessionId, botId, { isLoading: false });
                },
                (error) => {
                    const errorMessage = error.message || "알 수 없는 오류가 발생했습니다.";
                    chatStore.updateMessage(sessionId, botId, {
                        content: (chatStore.sessions[sessionId]?.messages.find(m => m.id === botId)?.content || "") + `\n\n⚠️ 오류: ${errorMessage}`,
                        isLoading: false
                    });
                    setIsLoading(false);
                }
            );
        } catch (error) {
            const errorMessage =
                error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다.";

            chatStore.updateMessage(sessionId, botId, {
                content: `⚠️ 오류: ${errorMessage}`,
                isLoading: false
            });
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full overflow-hidden">
            {/* Header */}
            <div className="bg-background/95 backdrop-blur p-3 border-b border-gray-200 dark:border-zinc-800">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                        <Link
                            href="/documents"
                            className="p-2 rounded-lg hover:bg-white/10 transition text-gray-400 hover:text-white"
                        >
                            <ArrowLeft size={20} />
                        </Link>
                        <div className="flex items-center space-x-2">
                            <div className="w-8 h-8 bg-blue-500/10 rounded-lg flex items-center justify-center">
                                <FileText size={16} className="text-blue-400" />
                            </div>
                            <div>
                                <h1 className="text-sm font-semibold text-white">{filename}</h1>
                                <p className="text-xs text-gray-500">문서 내 질문-답변</p>
                            </div>
                        </div>
                    </div>
                    <ModelSelector value={provider} onChange={setProvider} disabled={isLoading} />
                </div>

                {/* 검색 확장 체크박스 */}
                <div className="flex items-center gap-2 pl-12">
                    <input
                        type="checkbox"
                        id="expand-search"
                        checked={expand}
                        onChange={(e) => setExpand(e.target.checked)}
                        disabled={isLoading}
                        className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                    />
                    <label
                        htmlFor="expand-search"
                        className="text-sm font-medium text-muted-foreground cursor-pointer select-none flex items-center gap-1"
                    >
                        🔍 검색어 확장(번역)
                    </label>
                </div>
            </div>

            <ChatList messages={messages} />
            <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
        </div>
    );
}
