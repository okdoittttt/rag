"use client";

import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { ChatList, ChatInput, Message } from "@/components/chat";
import { ModelSelector } from "@/components/chat/ModelSelector";
import { askQuestionStream } from "@/lib/api";
import { useSettingsStore } from "@/lib/store";
import { useChatStore } from "@/lib/chatStore";

export default function Home() {
  const { data: session } = useSession();
  const settings = useSettingsStore();
  const chatStore = useChatStore();

  const [isLoading, setIsLoading] = useState(false);
  const [provider, setProvider] = useState<"gemini" | "ollama">("gemini");
  const [mounted, setMounted] = useState(false);

  // Only set mounted state, don't auto-create sessions
  useEffect(() => {
    setMounted(true);
  }, []);

  const currentSession = chatStore.getCurrentSession();
  const messages = mounted && currentSession ? currentSession.messages : [];

  const handleSubmit = async (query: string) => {
    // Create session on first message if none exists
    let sessionId = chatStore.currentSessionId;
    if (!sessionId) {
      sessionId = chatStore.createSession();
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query,
    };
    chatStore.addMessage(sessionId, userMessage);

    // Add loading bot message
    const botId = (Date.now() + 1).toString();
    const botMessage: Message = { id: botId, role: "assistant", content: "", isLoading: true };
    chatStore.addMessage(sessionId, botMessage);

    setIsLoading(true);

    try {
      // Call actual API with selected provider and user_id (Streaming)
      await askQuestionStream(
        query,
        {
          provider,
          user_id: session?.user?.id,
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
      // Setup error handling (fallback)
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
      <div className="bg-background/95 backdrop-blur p-2 flex justify-start sticky top-0 z-10">
        <ModelSelector value={provider} onChange={setProvider} disabled={isLoading} />
      </div>
      <ChatList messages={messages} />
      <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
    </div>
  );
}
