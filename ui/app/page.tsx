"use client";

import { useState } from "react";
import { ChatList, ChatInput, Message } from "@/components/chat";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

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

    // TODO: Replace with actual API call in Phase 3
    // Mock response for now
    setTimeout(() => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botId
            ? {
              ...m,
              content:
                "이것은 모의 응답입니다. Phase 3에서 실제 API와 연동됩니다.\n\n**Markdown**도 지원합니다:\n- 리스트 항목 1\n- 리스트 항목 2\n\n```python\nprint('Hello, RAG!')\n```",
              isLoading: false,
              references: [
                { content: "샘플 참조 1...", source: "document1.md", score: 0.92 },
                { content: "샘플 참조 2...", source: "document2.md", score: 0.85 },
              ],
            }
            : m
        )
      );
      setIsLoading(false);
    }, 1500);
  };

  return (
    <div className="flex flex-col flex-1">
      <ChatList messages={messages} />
      <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
    </div>
  );
}
