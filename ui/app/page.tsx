"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import { ChatList, ChatInput, Message } from "@/components/chat";
import { ModelSelector } from "@/components/chat/ModelSelector";
import { askQuestionStream } from "@/lib/api";

export default function Home() {
  const { data: session } = useSession();
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
      // Call actual API with selected provider and user_id (Streaming)
      await askQuestionStream(
        query,
        { provider, user_id: session?.user?.id },
        (text) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botId ? { ...m, content: m.content + text } : m
            )
          );
        },
        (refs) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === botId ? { ...m, references: refs } : m))
          );
        },
        () => {
          setIsLoading(false);
          setMessages((prev) =>
            prev.map((m) => (m.id === botId ? { ...m, isLoading: false } : m))
          );
        },
        (error) => {
          const errorMessage = error.message || "알 수 없는 오류가 발생했습니다.";
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botId
                ? {
                  ...m,
                  content: m.content + `\n\n⚠️ 오류: ${errorMessage}`,
                  isLoading: false,
                }
                : m
            )
          );
          setIsLoading(false);
        }
      );
    } catch (error) {
      // Setup error handling (fallback)
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
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col flex-1">
      <div className="bg-background/95 backdrop-blur p-2 flex justify-start sticky top-0 z-10">
        <ModelSelector value={provider} onChange={setProvider} disabled={isLoading} />
      </div>
      <ChatList messages={messages} />
      <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
    </div>
  );
}
