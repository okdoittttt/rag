import React from "react";
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
    const [selectedRef, setSelectedRef] = React.useState<{ content: string; source: string; score: number } | null>(null);

    // Helper to extract filename from path
    const getFileName = (path: string) => {
        return path.split(/[/\\]/).pop() || path;
    };

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
                                            <button
                                                key={idx}
                                                onClick={() => setSelectedRef(ref)}
                                                className="text-xs px-2 py-1 bg-muted rounded border hover:bg-muted/80 transition-colors flex items-center gap-1 group"
                                            >
                                                <span className="max-w-[150px] truncate text-muted-foreground group-hover:text-foreground">
                                                    {getFileName(ref.source)}
                                                </span>
                                                <span className="text-indigo-500 font-medium">
                                                    {(ref.score * 100).toFixed(0)}%
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>

            {/* Reference Data Modal */}
            {selectedRef && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm" onClick={() => setSelectedRef(null)}>
                    <div
                        className="bg-background border rounded-lg shadow-lg w-full max-w-lg overflow-hidden flex flex-col max-h-[80vh]"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="p-4 border-b flex items-center justify-between bg-muted/20">
                            <div className="flex items-center gap-2 font-medium truncate">
                                <FileText size={16} className="text-muted-foreground" />
                                <span className="truncate">{getFileName(selectedRef.source)}</span>
                            </div>
                            <span className="text-xs bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300 px-2 py-0.5 rounded-full font-medium">
                                유사도 {(selectedRef.score * 100).toFixed(0)}%
                            </span>
                        </div>
                        <div className="p-4 overflow-y-auto bg-muted/10 text-sm leading-relaxed whitespace-pre-wrap">
                            {selectedRef.content}
                            {selectedRef.content.length >= 500 && (
                                <p className="mt-4 text-xs text-muted-foreground italic">(일부 내용만 표시됨)</p>
                            )}
                        </div>
                        <div className="p-3 border-t bg-muted/20 flex justify-end">
                            <button
                                onClick={() => setSelectedRef(null)}
                                className="px-4 py-2 text-sm font-medium bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
                            >
                                닫기
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
