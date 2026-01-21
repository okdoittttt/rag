"use client";

import FileUpload from "@/components/upload/FileUpload";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function UploadPage() {
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
            <div className="flex-1 flex flex-col items-center justify-center px-4">
                <div className="w-full max-w-xl">
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
                            <strong className="text-gray-300">지원 형식:</strong> .txt, .md, .pdf
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
