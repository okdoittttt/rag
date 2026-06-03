"use client";

import { useEffect, useState } from "react";
import FileUpload from "@/components/upload/FileUpload";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";

const FALLBACK_EXTENSIONS = [".txt", ".md", ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".hwpx"];

export default function UploadPage() {
    const [supportedExtensions, setSupportedExtensions] = useState<string[]>(FALLBACK_EXTENSIONS);

    // 지원 확장자 목록을 백엔드(파서 레지스트리)에서 가져온다.
    useEffect(() => {
        let active = true;
        fetch("/api/supported-extensions")
            .then((res) => res.json())
            .then((data) => {
                if (active && Array.isArray(data.extensions) && data.extensions.length) {
                    setSupportedExtensions(data.extensions);
                }
            })
            .catch(() => {
                // 조회 실패 시 폴백 목록 유지
            });
        return () => {
            active = false;
        };
    }, []);

    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <div className="p-4 flex items-center space-x-3">
                <Link
                    href="/"
                    className="p-2 rounded-lg hover:bg-white/10 transition text-gray-400 hover:text-white"
                >
                    <ArrowLeft size={20} />
                </Link>
                <h1 className="text-lg font-semibold text-white">문서 업로드</h1>
            </div>

            {/* Content */}
            <div className="flex-1 flex flex-col items-center px-4 overflow-y-auto">
                <div className="w-full max-w-xl my-auto py-10">
                    <div className="text-center mb-8">
                        <h2 className="text-2xl font-bold text-white mb-2">
                            RAG 문서 업로드
                        </h2>
                        <p className="text-gray-400">
                            업로드한 문서는 자동으로 청킹되어 검색 가능한 형태로 저장됩니다.
                        </p>
                    </div>

                    <FileUpload />

                    <div className="mt-6 p-4 bg-white/5 rounded-lg border border-white/10">
                        <p className="text-sm text-gray-400">
                            <strong className="text-gray-300">지원 형식:</strong> {supportedExtensions.join(", ")}
                        </p>
                        <p className="text-sm text-gray-400 mt-1">
                            <strong className="text-gray-300">격리된 저장:</strong> 업로드한 문서는 본인 계정에서만 검색됩니다.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
