import ReactMarkdown from "react-markdown";
import { User, Bot, FileText } from "lucide-react";

export interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    references?: {
        content: string;
        source: string;
        score: number;
    }[];
    isLoading?: boolean;
}

interface ChatMessageProps {
    message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === "user";

    return (
        <div className={`py-4 ${isUser ? "bg-transparent" : "bg-muted/30"}`}>
            <div className="max-w-3xl mx-auto px-4 flex gap-4">
                {/* Avatar */}
                <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${isUser ? "bg-blue-600" : "bg-green-600"
                        }`}
                >
                    {isUser ? (
                        <User size={18} className="text-white" />
                    ) : (
                        <Bot size={18} className="text-white" />
                    )}
                </div>

                {/* Content */}
                <div className="flex-1 space-y-3 overflow-hidden">
                    {/* Loading state */}
                    {message.isLoading ? (
                        <div className="flex gap-1">
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                        </div>
                    ) : (
                        <>
                            {/* Message content */}
                            <div className="prose prose-sm dark:prose-invert max-w-none">
                                {isUser ? (
                                    <p>{message.content}</p>
                                ) : (
                                    <ReactMarkdown>{message.content}</ReactMarkdown>
                                )}
                            </div>

                            {/* References (Bot only) */}
                            {!isUser && message.references && message.references.length > 0 && (
                                <div className="mt-4 pt-4 border-t">
                                    <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
                                        <FileText size={12} />
                                        참조 문서
                                    </p>
                                    <div className="flex flex-wrap gap-2">
                                        {message.references.map((ref, idx) => (
                                            <div
                                                key={idx}
                                                className="text-xs px-2 py-1 bg-muted rounded border cursor-pointer hover:bg-muted/80"
                                                title={ref.content}
                                            >
                                                {ref.source} ({(ref.score * 100).toFixed(0)}%)
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
