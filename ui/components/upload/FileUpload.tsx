"use client";

import { useState, useCallback } from "react";
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader2, Play } from "lucide-react";

interface FileUploadProps {
    onUploadComplete?: () => void;
}

interface UploadStatus {
    file: File;
    status: "pending" | "uploading" | "success" | "error";
    message?: string;
    chunkCount?: number;
}

export default function FileUpload({ onUploadComplete }: FileUploadProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [uploads, setUploads] = useState<UploadStatus[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const uploadFile = async (file: File): Promise<{ success: boolean; message: string; chunkCount?: number }> => {
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                return { success: false, message: data.error || "업로드 실패" };
            }

            return { success: true, message: data.message, chunkCount: data.chunk_count };
        } catch (error) {
            return { success: false, message: error instanceof Error ? error.message : "네트워크 오류" };
        }
    };

    const addFiles = (files: FileList | File[]) => {
        const fileArray = Array.from(files);
        const allowedTypes = [".txt", ".md", ".pdf"];

        // 중복 파일 제거
        const existingNames = new Set(uploads.map(u => u.file.name));

        const validFiles = fileArray.filter((file) => {
            const ext = "." + file.name.split(".").pop()?.toLowerCase();
            if (!allowedTypes.includes(ext)) return false;
            if (existingNames.has(file.name)) return false;
            return true;
        });

        if (validFiles.length === 0 && fileArray.length > 0) {
            // 이미 추가된 파일이거나 지원하지 않는 형식인 경우 조용히 무시하거나 알림
            return;
        }

        const newUploads: UploadStatus[] = validFiles.map((file) => ({
            file,
            status: "pending",
        }));

        setUploads((prev) => [...prev, ...newUploads]);
    };

    const handleStartUpload = async () => {
        setIsProcessing(true);
        const pendingFiles = uploads.filter(u => u.status === "pending" || u.status === "error");

        if (pendingFiles.length === 0) {
            setIsProcessing(false);
            return;
        }

        // 전체 업로드 목록에서 인덱스 찾아서 업데이트
        for (let i = 0; i < uploads.length; i++) {
            if (uploads[i].status !== "pending" && uploads[i].status !== "error") continue;

            setUploads((prev) =>
                prev.map((u, idx) => (idx === i ? { ...u, status: "uploading" } : u))
            );

            const result = await uploadFile(uploads[i].file);

            setUploads((prev) =>
                prev.map((u, idx) =>
                    idx === i
                        ? {
                            ...u,
                            status: result.success ? "success" : "error",
                            message: result.message,
                            chunkCount: result.chunkCount,
                        }
                        : u
                )
            );
        }

        setIsProcessing(false);
        onUploadComplete?.();
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        addFiles(e.dataTransfer.files);
    }, [uploads]);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            addFiles(e.target.files);
            // 같은 파일 다시 선택 가능하게 초기화
            e.target.value = "";
        }
    };

    const clearUpload = (index: number) => {
        if (isProcessing) return;
        setUploads((prev) => prev.filter((_, i) => i !== index));
    };

    const clearAll = () => {
        if (isProcessing) return;
        setUploads([]);
    };

    // 업로드 대상(대기중 또는 실패)이 있는지 확인
    const hasPendingFiles = uploads.some(u => u.status === "pending" || u.status === "error");

    return (
        <div className="w-full max-w-xl mx-auto p-4">
            {/* Drop Zone */}
            <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all ${isDragging
                    ? "border-blue-500 bg-blue-500/10"
                    : "border-white/20 hover:border-white/40 bg-white/5"
                    }`}
            >
                <input
                    type="file"
                    multiple
                    accept=".txt,.md,.pdf"
                    onChange={handleFileSelect}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    disabled={isProcessing}
                />
                <Upload
                    size={40}
                    className={`mx-auto mb-4 ${isDragging ? "text-blue-400" : "text-gray-400"}`}
                />
                <p className="text-white font-medium mb-1">
                    파일을 드래그하거나 클릭하여 선택
                </p>
                <p className="text-gray-500 text-sm">.txt, .md, .pdf 파일 지원</p>
            </div>

            {/* Action Buttons */}
            {uploads.length > 0 && (
                <div className="mt-4 flex items-center justify-between">
                    <div className="text-sm text-gray-400">
                        총 {uploads.length}개 파일
                    </div>
                    <div className="flex space-x-2">
                        <button
                            onClick={clearAll}
                            disabled={isProcessing}
                            className="px-3 py-1.5 text-sm text-gray-400 hover:text-white transition disabled:opacity-50"
                        >
                            모두 지우기
                        </button>
                        {hasPendingFiles && (
                            <button
                                onClick={handleStartUpload}
                                disabled={isProcessing}
                                className="flex items-center space-x-1 px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isProcessing ? (
                                    <>
                                        <Loader2 size={16} className="animate-spin" />
                                        <span>처리중...</span>
                                    </>
                                ) : (
                                    <>
                                        <Play size={16} />
                                        <span>파싱 시작</span>
                                    </>
                                )}
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* Upload List */}
            {uploads.length > 0 && (
                <div className="mt-2 space-y-2">
                    {uploads.map((upload, index) => (
                        <div
                            key={`${upload.file.name}-${index}`}
                            className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10"
                        >
                            <div className="flex items-center space-x-3 overflow-hidden">
                                <FileText size={18} className="text-gray-400 flex-shrink-0" />
                                <div className="overflow-hidden">
                                    <p className="text-sm text-white truncate">{upload.file.name}</p>
                                    {upload.status === "success" && upload.chunkCount && (
                                        <p className="text-xs text-green-400">{upload.chunkCount}개 청크 생성</p>
                                    )}
                                    {upload.status === "error" && (
                                        <p className="text-xs text-red-400">{upload.message}</p>
                                    )}
                                    {upload.status === "pending" && (
                                        <p className="text-xs text-gray-500">대기중</p>
                                    )}
                                </div>
                            </div>

                            <div className="flex items-center space-x-2">
                                {upload.status === "uploading" && (
                                    <Loader2 size={18} className="text-blue-400 animate-spin" />
                                )}
                                {upload.status === "success" && (
                                    <CheckCircle size={18} className="text-green-400" />
                                )}
                                {upload.status === "error" && (
                                    <AlertCircle size={18} className="text-red-400" />
                                )}
                                <button
                                    onClick={() => clearUpload(index)}
                                    disabled={isProcessing}
                                    className="p-1 text-gray-500 hover:text-white transition disabled:opacity-50"
                                >
                                    <X size={14} />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
