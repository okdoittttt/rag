"use client";

import { useEffect, useState } from "react";
import { FileText, Trash2, ArrowLeft, Search, Loader2, MessageSquare, CheckSquare, Square, MoreVertical } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface Document {
    id: string;
    filename: string;
    filesize: number;
    chunkCount: number;
    createdAt: string;
}

function formatFileSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleDateString("ko-KR", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });
}

export default function DocumentsPage() {
    const router = useRouter();
    const [documents, setDocuments] = useState<Document[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [isDeleting, setIsDeleting] = useState(false);

    const fetchDocuments = async () => {
        setIsLoading(true);
        try {
            const response = await fetch("/api/upload");
            if (response.ok) {
                const data = await response.json();
                setDocuments(data.documents || []);
            }
        } catch (error) {
            console.error("Failed to fetch documents:", error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchDocuments();
    }, []);

    const filteredDocuments = documents.filter((doc) =>
        doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const toggleSelection = (id: string) => {
        const newSelected = new Set(selectedIds);
        if (newSelected.has(id)) {
            newSelected.delete(id);
        } else {
            newSelected.add(id);
        }
        setSelectedIds(newSelected);
    };

    const toggleAll = () => {
        if (selectedIds.size === filteredDocuments.length && filteredDocuments.length > 0) {
            setSelectedIds(new Set());
        } else {
            setSelectedIds(new Set(filteredDocuments.map((d) => d.id)));
        }
    };

    const handleBulkDelete = async () => {
        if (selectedIds.size === 0) return;

        if (!confirm(`선택한 ${selectedIds.size}개의 문서를 삭제하시겠습니까? 인덱싱된 청크도 함께 삭제됩니다.`)) {
            return;
        }

        setIsDeleting(true);
        try {
            // 병렬로 삭제 요청
            const deletePromises = Array.from(selectedIds).map((id) =>
                fetch(`/api/documents/${id}`, { method: "DELETE" })
            );

            await Promise.all(deletePromises);

            // 목록 갱신 (성공한 것만 제거하는게 좋지만, 간단히 전체 리프레시 혹은 필터링)
            setDocuments((prev) => prev.filter((d) => !selectedIds.has(d.id)));
            setSelectedIds(new Set());
        } catch (error) {
            console.error("Bulk delete error:", error);
            alert("일부 문서를 삭제하는 중 오류가 발생했습니다.");
            // 실패 시 최신 상태를 위해 목록 다시 불러오기
            fetchDocuments();
        } finally {
            setIsDeleting(false);
        }
    };

    const handleBulkChat = () => {
        if (selectedIds.size === 0) return;

        if (selectedIds.size === 1) {
            // 단일 문서 채팅: 기존 라우트 사용
            const docId = Array.from(selectedIds)[0];
            const doc = documents.find(d => d.id === docId);
            if (doc) {
                router.push(`/documents/${docId}/chat?filename=${encodeURIComponent(doc.filename)}`);
            }
        } else {
            // 다중 문서 채팅 (구현 예정)
            alert("다중 문서 채팅 기능은 준비 중입니다. 현재는 한 번에 하나의 문서만 선택하여 채팅할 수 있습니다.");
            // 추후 구현 시: router.push(`/chat?docs=${Array.from(selectedIds).join(",")}`);
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="p-4 flex items-center justify-between border-b border-white/10 shrink-0">
                <div className="flex items-center space-x-3">
                    <Link
                        href="/"
                        className="p-2 rounded-lg hover:bg-white/10 transition text-gray-400 hover:text-white"
                    >
                        <ArrowLeft size={20} />
                    </Link>
                    <h1 className="text-lg font-semibold text-white">내 문서</h1>
                    <span className="text-sm text-gray-500">({documents.length}개)</span>
                </div>

                <div className="flex items-center space-x-2">
                    {selectedIds.size > 0 && (
                        <>
                            <button
                                onClick={handleBulkChat}
                                className="px-3 py-2 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 text-sm font-medium rounded-lg transition flex items-center space-x-2"
                            >
                                <MessageSquare size={16} />
                                <span>채팅 ({selectedIds.size})</span>
                            </button>
                            <button
                                onClick={handleBulkDelete}
                                disabled={isDeleting}
                                className="px-3 py-2 bg-red-600/10 hover:bg-red-600/20 text-red-400 text-sm font-medium rounded-lg transition flex items-center space-x-2"
                            >
                                {isDeleting ? <Loader2 size={16} className="animate-spin" /> : <Trash2 size={16} />}
                                <span>삭제 ({selectedIds.size})</span>
                            </button>
                            <div className="w-px h-6 bg-white/10 mx-2" />
                        </>
                    )}
                    <Link
                        href="/upload"
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition"
                    >
                        문서 업로드
                    </Link>
                </div>
            </div>

            {/* Search */}
            <div className="p-4 shrink-0">
                <div className="relative">
                    <Search
                        size={18}
                        className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"
                    />
                    <input
                        type="text"
                        placeholder="문서 검색..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 placeholder-gray-500"
                    />
                </div>
            </div>

            {/* Document List */}
            <div className="flex-1 overflow-y-auto px-4 pb-4">
                {isLoading ? (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 size={24} className="animate-spin text-gray-500" />
                    </div>
                ) : filteredDocuments.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                        {searchQuery ? "검색 결과가 없습니다." : "아직 업로드한 문서가 없습니다."}
                    </div>
                ) : (
                    <>
                        {/* List Actions / Select All */}
                        <div className="flex items-center px-4 py-2 mb-2">
                            <button
                                onClick={toggleAll}
                                className="flex items-center space-x-2 text-sm text-gray-400 hover:text-white transition"
                            >
                                {selectedIds.size > 0 && selectedIds.size === filteredDocuments.length ? (
                                    <CheckSquare size={18} className="text-blue-500" />
                                ) : (
                                    <Square size={18} />
                                )}
                                <span>전체 선택</span>
                            </button>
                        </div>

                        <div className="space-y-2">
                            {filteredDocuments.map((doc) => {
                                const isSelected = selectedIds.has(doc.id);
                                return (
                                    <div
                                        key={doc.id}
                                        onClick={() => toggleSelection(doc.id)}
                                        className={`flex items-center justify-between p-4 rounded-xl border transition cursor-pointer group ${isSelected
                                                ? "bg-blue-500/10 border-blue-500/30"
                                                : "bg-white/5 border-white/10 hover:bg-white/10"
                                            }`}
                                    >
                                        <div className="flex items-center space-x-4 overflow-hidden">
                                            <div
                                                className={`flex-shrink-0 transition-colors ${isSelected ? "text-blue-500" : "text-gray-500 group-hover:text-gray-400"}`}
                                            >
                                                {isSelected ? <CheckSquare size={20} /> : <Square size={20} />}
                                            </div>

                                            <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                                                <FileText size={20} className="text-blue-400" />
                                            </div>
                                            <div className="overflow-hidden">
                                                <p className={`text-sm font-medium truncate ${isSelected ? "text-blue-200" : "text-white"}`}>
                                                    {doc.filename}
                                                </p>
                                                <p className="text-xs text-gray-500">
                                                    {formatFileSize(doc.filesize)} · {doc.chunkCount}개 청크 ·{" "}
                                                    {formatDate(doc.createdAt)}
                                                </p>
                                            </div>
                                        </div>

                                        {/* Individual Actions (Only visible on hover if not selected mode? Or always keep them?) 
                                            Let's keep them but make them prevent propagation to avoid toggling selection
                                        */}
                                        <div className="flex items-center space-x-2" onClick={(e) => e.stopPropagation()}>
                                            <button
                                                onClick={() => router.push(`/documents/${doc.id}/chat?filename=${encodeURIComponent(doc.filename)}`)}
                                                className="p-2 text-gray-500 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition opacity-0 group-hover:opacity-100"
                                                title="이 문서로 채팅"
                                            >
                                                <MessageSquare size={16} />
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
