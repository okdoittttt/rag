"use client";

import { useState, useCallback } from "react";
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader2 } from "lucide-react";

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

    const processFiles = async (files: FileList | File[]) => {
        const fileArray = Array.from(files);
        const allowedTypes = [".txt", ".md"];

        const validFiles = fileArray.filter((file) => {
            const ext = "." + file.name.split(".").pop()?.toLowerCase();
            return allowedTypes.includes(ext);
        });

        if (validFiles.length === 0) {
            alert("지원되는 파일 형식: .txt, .md");
            return;
        }

        // 초기 상태 설정
        const initialStatuses: UploadStatus[] = validFiles.map((file) => ({
            file,
            status: "pending",
        }));
        setUploads(initialStatuses);

        // 순차적으로 업로드
        for (let i = 0; i < validFiles.length; i++) {
            setUploads((prev) =>
                prev.map((u, idx) => (idx === i ? { ...u, status: "uploading" } : u))
            );

            const result = await uploadFile(validFiles[i]);

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

        onUploadComplete?.();
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        processFiles(e.dataTransfer.files);
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            processFiles(e.target.files);
        }
    };

    const clearUpload = (index: number) => {
        setUploads((prev) => prev.filter((_, i) => i !== index));
    };

    const clearAll = () => {
        setUploads([]);
    };

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
                    accept=".txt,.md"
                    onChange={handleFileSelect}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <Upload
                    size={40}
                    className={`mx-auto mb-4 ${isDragging ? "text-blue-400" : "text-gray-400"}`}
                />
                <p className="text-white font-medium mb-1">
                    파일을 드래그하거나 클릭하여 업로드
                </p>
                <p className="text-gray-500 text-sm">.txt, .md 파일 지원</p>
            </div>

            {/* Upload List */}
            {uploads.length > 0 && (
                <div className="mt-4 space-y-2">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-400">
                            {uploads.filter((u) => u.status === "success").length} / {uploads.length} 완료
                        </span>
                        <button
                            onClick={clearAll}
                            className="text-xs text-gray-500 hover:text-white transition"
                        >
                            모두 지우기
                        </button>
                    </div>

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
                                </div>
                            </div>

                            <div className="flex items-center space-x-2">
                                {upload.status === "pending" && (
                                    <span className="text-xs text-gray-500">대기중</span>
                                )}
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
                                    className="p-1 text-gray-500 hover:text-white transition"
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
