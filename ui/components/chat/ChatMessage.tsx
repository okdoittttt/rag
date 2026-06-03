import React from "react";
import ReactMarkdown from "react-markdown";
import { User, Bot, FileText, Brain, ChevronDown, ChevronRight } from "lucide-react";

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
    reasoning?: string;  // CoT 추론 과정(스트리밍 누적)
    status?: string;     // 진행 단계 phase (searching/analyzing/reasoning_start)
}

interface ChatMessageProps {
    message: Message;
}

/** 진행 단계(phase)를 사용자에게 보여줄 한국어 라벨로 변환한다. */
function statusLabel(status?: string): string {
    switch (status) {
        case "searching":
            return "🔎 관련 문서를 검색하고 있어요…";
        case "analyzing":
            return "📖 문서를 분석하고 있어요…";
        case "reasoning_start":
            return "💭 생각하고 있어요…";
        default:
            return "처리하고 있어요…";
    }
}

/** CoT 추론 과정을 접이식으로 표시하는 블록. */
function ReasoningBlock({
    reasoning,
    streaming,
    open,
    onToggle,
}: {
    reasoning: string;
    streaming: boolean;
    open: boolean;
    onToggle: () => void;
}) {
    return (
        <div className="rounded-lg border border-white/10 bg-muted/40 text-sm overflow-hidden">
            <button
                onClick={onToggle}
                className="flex items-center gap-2 w-full px-3 py-2 text-left text-muted-foreground hover:text-foreground transition"
            >
                {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                <Brain size={14} className={streaming ? "animate-pulse text-indigo-400" : "text-indigo-400"} />
                <span>{streaming ? "생각 중…" : "추론 과정 보기"}</span>
            </button>
            {open && (
                <div className="px-3 pb-3 whitespace-pre-wrap text-muted-foreground leading-relaxed">
                    {reasoning}
                </div>
            )}
        </div>
    );
}

export function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.role === "user";
    const [selectedRef, setSelectedRef] = React.useState<{ content: string; source: string; score: number } | null>(null);
    const [showReasoning, setShowReasoning] = React.useState(true);
    // 참조 문서 섹션 전체 접힘 상태(기본 접힘) 및 펼쳐진 문서 그룹 키 집합.
    const [showRefs, setShowRefs] = React.useState(false);
    const [expandedGroups, setExpandedGroups] = React.useState<Set<string>>(new Set());

    /** 특정 문서 그룹의 펼침/접힘을 토글한다. */
    const toggleGroup = (key: string) => {
        setExpandedGroups((prev) => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    const hasContent = !!message.content;
    const hasReasoning = !!message.reasoning;
    // 답변이 비었는데 추론만 있고 스트림이 끝났으면(모델이 구분자 형식을 무시한 경우)
    // 추론을 최종 답변으로 승격해 표시한다.
    const promotedContent = !hasContent && !message.isLoading && hasReasoning ? message.reasoning : null;

    // 답변 토큰이 도착하기 시작하면 추론 블록을 자동으로 접는다(수동 토글 가능).
    React.useEffect(() => {
        if (hasContent) setShowReasoning(false);
    }, [hasContent]);

    // Helper to extract filename from path
    const getFileName = (path: string) => {
        return path.split(/[/\\]/).pop() || path;
    };

    // 참조 청크를 source(문서) 단위로 그룹핑한다. 청크는 중복 제거하지 않고
    // 같은 문서로 묶어, "버튼 수십 개" 대신 "문서 N개"로 체감 부담을 줄인다.
    // 그룹은 최고 점수 내림차순, 그룹 내 청크도 점수 내림차순으로 정렬한다.
    const referenceGroups = React.useMemo(() => {
        const refs = message.references ?? [];
        const map = new Map<string, { name: string; chunks: typeof refs }>();
        for (const ref of refs) {
            const key = ref.source;
            if (!map.has(key)) map.set(key, { name: getFileName(ref.source), chunks: [] });
            map.get(key)!.chunks.push(ref);
        }
        return Array.from(map.entries())
            .map(([key, g]) => ({
                key,
                name: g.name,
                chunks: [...g.chunks].sort((a, b) => b.score - a.score),
                bestScore: Math.max(...g.chunks.map((c) => c.score)),
            }))
            .sort((a, b) => b.bestScore - a.bestScore);
    }, [message.references]);

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
                    {isUser ? (
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                            <p>{message.content}</p>
                        </div>
                    ) : (
                        <>
                            {/* CoT 추론 과정 (답변으로 승격된 경우는 제외) */}
                            {hasReasoning && !promotedContent && (
                                <ReasoningBlock
                                    reasoning={message.reasoning ?? ""}
                                    streaming={!!message.isLoading && !hasContent}
                                    open={showReasoning}
                                    onToggle={() => setShowReasoning((v) => !v)}
                                />
                            )}

                            {/* 최종 답변 (스트리밍 중에도 실시간 렌더) */}
                            {(hasContent || promotedContent) && (
                                <div className="prose prose-sm dark:prose-invert max-w-none">
                                    <ReactMarkdown>{message.content || promotedContent || ""}</ReactMarkdown>
                                </div>
                            )}

                            {/* 아직 추론/답변 토큰이 없을 때: 진행 단계 라벨 또는 점 애니메이션 */}
                            {message.isLoading && !hasContent && !hasReasoning && (
                                message.status ? (
                                    <p className="text-sm text-muted-foreground animate-pulse">
                                        {statusLabel(message.status)}
                                    </p>
                                ) : (
                                    <div className="flex gap-1">
                                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                                    </div>
                                )
                            )}

                            {/* References (Bot only) — 문서 단위 그룹핑 + 전체 접기 */}
                            {message.references && message.references.length > 0 && (
                                <div className="mt-4 pt-4 border-t">
                                    <button
                                        onClick={() => setShowRefs((v) => !v)}
                                        className="text-xs text-muted-foreground mb-2 flex items-center gap-1 hover:text-foreground transition-colors"
                                    >
                                        {showRefs ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                                        <FileText size={12} />
                                        참조 문서 {message.references.length}개
                                        <span className="text-muted-foreground/60">· 문서 {referenceGroups.length}개</span>
                                    </button>

                                    {showRefs && (
                                        <div className="flex flex-col gap-1.5">
                                            {referenceGroups.map((group) => {
                                                const expanded = expandedGroups.has(group.key);
                                                return (
                                                    <div key={group.key} className="rounded border bg-muted/40 overflow-hidden">
                                                        <button
                                                            onClick={() => toggleGroup(group.key)}
                                                            className="flex items-center gap-1.5 w-full px-2 py-1.5 text-left text-xs hover:bg-muted/70 transition-colors"
                                                        >
                                                            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                                                            <span className="flex-1 truncate text-muted-foreground">{group.name}</span>
                                                            <span className="text-muted-foreground/70 shrink-0">청크 {group.chunks.length}개</span>
                                                            <span className="text-indigo-500 font-medium shrink-0">
                                                                최고 {(group.bestScore * 100).toFixed(0)}%
                                                            </span>
                                                        </button>
                                                        {expanded && (
                                                            <div className="flex flex-wrap gap-2 px-2 pb-2 pt-1">
                                                                {group.chunks.map((ref, idx) => (
                                                                    <button
                                                                        key={idx}
                                                                        onClick={() => setSelectedRef(ref)}
                                                                        className="text-xs px-2 py-1 bg-background rounded border hover:bg-muted/80 transition-colors flex items-center gap-1 group"
                                                                    >
                                                                        <span className="text-muted-foreground group-hover:text-foreground">
                                                                            청크 {idx + 1}
                                                                        </span>
                                                                        <span className="text-indigo-500 font-medium">
                                                                            {(ref.score * 100).toFixed(0)}%
                                                                        </span>
                                                                    </button>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
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
