"use client";

import { useEffect, useRef } from "react";
import { ChatMessage, Message } from "./ChatMessage";

interface ChatListProps {
    messages: Message[];
}

export function ChatList({ messages }: ChatListProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    if (messages.length === 0) {
        return <div className="flex-1" />;
    }

    return (
        <div className="flex-1 overflow-y-auto">
            {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
            ))}
            <div ref={bottomRef} />
        </div>
    );
}
