"use client";

import { useState, KeyboardEvent } from "react";
import { Send } from "lucide-react";

interface ChatInputProps {
    onSubmit: (query: string) => void;
    isLoading?: boolean;
}

export function ChatInput({ onSubmit, isLoading = false }: ChatInputProps) {
    const [input, setInput] = useState("");

    const handleSubmit = () => {
        const trimmed = input.trim();
        if (!trimmed || isLoading) return;
        onSubmit(trimmed);
        setInput("");
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="border-t p-4 bg-background">
            <div className="max-w-3xl mx-auto flex gap-2 items-end">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="질문을 입력하세요..."
                    rows={1}
                    disabled={isLoading}
                    className="flex-1 resize-none rounded-lg border px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 bg-background"
                />
                <button
                    onClick={handleSubmit}
                    disabled={!input.trim() || isLoading}
                    className="p-3 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                    <Send size={20} />
                </button>
            </div>
        </div>
    );
}
