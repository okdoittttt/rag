"use client";

import { useEffect, useState } from "react";
import { FileText, Trash2, ArrowLeft, Search, Loader2 } from "lucide-react";
import Link from "next/link";

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
    const [documents, setDocuments] = useState<Document[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const [deletingId, setDeletingId] = useState<string | null>(null);

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

    const handleDelete = async (id: string) => {
        if (!confirm("이 문서를 삭제하시겠습니까? 인덱싱된 청크도 함께 삭제됩니다.")) {
            return;
        }

        setDeletingId(id);
        try {
            const response = await fetch(`/api/documents/${id}`, {
                method: "DELETE",
            });

            if (response.ok) {
                setDocuments((prev) => prev.filter((d) => d.id !== id));
            } else {
                alert("삭제에 실패했습니다.");
            }
        } catch (error) {
            console.error("Delete error:", error);
            alert("삭제 중 오류가 발생했습니다.");
        } finally {
            setDeletingId(null);
        }
    };

    const filteredDocuments = documents.filter((doc) =>
        doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="p-4 flex items-center justify-between border-b border-white/10">
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
                <Link
                    href="/upload"
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition"
                >
                    문서 업로드
                </Link>
            </div>

            {/* Search */}
            <div className="p-4">
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
                    <div className="space-y-2">
                        {filteredDocuments.map((doc) => (
                            <div
                                key={doc.id}
                                className="flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 rounded-xl border border-white/10 transition group"
                            >
                                <div className="flex items-center space-x-3 overflow-hidden">
                                    <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center flex-shrink-0">
                                        <FileText size={20} className="text-blue-400" />
                                    </div>
                                    <div className="overflow-hidden">
                                        <p className="text-sm font-medium text-white truncate">
                                            {doc.filename}
                                        </p>
                                        <p className="text-xs text-gray-500">
                                            {formatFileSize(doc.filesize)} · {doc.chunkCount}개 청크 ·{" "}
                                            {formatDate(doc.createdAt)}
                                        </p>
                                    </div>
                                </div>

                                <button
                                    onClick={() => handleDelete(doc.id)}
                                    disabled={deletingId === doc.id}
                                    className="p-2 text-gray-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition opacity-0 group-hover:opacity-100 disabled:opacity-50"
                                >
                                    {deletingId === doc.id ? (
                                        <Loader2 size={16} className="animate-spin" />
                                    ) : (
                                        <Trash2 size={16} />
                                    )}
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
