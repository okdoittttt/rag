import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";

const API_BASE_URL = process.env.API_BASE_URL || "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
    try {
        // 세션에서 user_id 추출
        const session = await auth();
        if (!session?.user?.id) {
            return NextResponse.json({ error: "인증이 필요합니다." }, { status: 401 });
        }

        const userId = session.user.id;

        // FormData에서 파일 추출
        const formData = await req.formData();
        const file = formData.get("file") as File | null;

        if (!file) {
            return NextResponse.json({ error: "파일이 없습니다." }, { status: 400 });
        }

        // 파일 확장자 검증
        const allowedExtensions = [".txt", ".md"];
        const ext = "." + file.name.split(".").pop()?.toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            return NextResponse.json(
                { error: "지원되지 않는 파일 형식입니다. (.txt, .md만 지원)" },
                { status: 400 }
            );
        }

        // 파일 내용 읽기
        const content = await file.text();

        if (!content.trim()) {
            return NextResponse.json(
                { error: "파일 내용이 비어있습니다." },
                { status: 400 }
            );
        }

        // 백엔드 /index API 호출
        const response = await fetch(`${API_BASE_URL}/index`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                content: content,
                filename: file.name,
                user_id: userId,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            return NextResponse.json(
                { error: errorData.detail || "인덱싱 실패" },
                { status: response.status }
            );
        }

        const data = await response.json();

        return NextResponse.json({
            message: data.message,
            chunk_count: data.chunk_count,
            filename: file.name,
        });
    } catch (error) {
        console.error("Upload error:", error);
        return NextResponse.json(
            { error: "서버 오류가 발생했습니다." },
            { status: 500 }
        );
    }
}
