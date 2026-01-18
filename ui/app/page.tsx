"use client";

import { useState } from "react";
import { ChatList, ChatInput, Message } from "@/components/chat";
import { ModelSelector } from "@/components/chat/ModelSelector";
import { askQuestion } from "@/lib/api";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [provider, setProvider] = useState<"gemini" | "ollama">("gemini");

  const handleSubmit = async (query: string) => {
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: query,
    };
    setMessages((prev) => [...prev, userMessage]);

    // Add loading bot message
    const botId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: botId, role: "assistant", content: "", isLoading: true },
    ]);
    setIsLoading(true);

    try {
      // Call actual API with selected provider
      const response = await askQuestion(query, { provider });

      setMessages((prev) =>
        prev.map((m) =>
          m.id === botId
            ? {
              ...m,
              content: response.answer,
              isLoading: false,
              references: response.references,
            }
            : m
        )
      );
    } catch (error) {
      // Handle error
      const errorMessage =
        error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다.";

      setMessages((prev) =>
        prev.map((m) =>
          m.id === botId
            ? {
              ...m,
              content: `⚠️ 오류: ${errorMessage}`,
              isLoading: false,
            }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col flex-1">
      <div className="border-b bg-background/95 backdrop-blur p-2 flex justify-center sticky top-0 z-10">
        <ModelSelector value={provider} onChange={setProvider} disabled={isLoading} />
      </div>
      <ChatList messages={messages} />
      <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
    </div>
  );
}
