import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { writeFile, mkdir } from "fs/promises";
import path from "path";

const API_BASE_URL = process.env.API_BASE_URL || "http://127.0.0.1:8000";
const UPLOAD_DIR = process.env.UPLOAD_DIR || "./data/uploads";

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
        const allowedExtensions = [".txt", ".md", ".pdf"];
        const ext = "." + file.name.split(".").pop()?.toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            return NextResponse.json(
                { error: "지원되지 않는 파일 형식입니다. (.txt, .md, .pdf만 지원)" },
                { status: 400 }
            );
        }

        // 파일 버퍼 읽기 (텍스트 변환 제거)
        const fileBuffer = Buffer.from(await file.arrayBuffer());

        // 사용자별 업로드 디렉토리 생성
        const userUploadDir = path.resolve(UPLOAD_DIR, userId);
        await mkdir(userUploadDir, { recursive: true });

        // 고유 파일명 생성 (타임스탬프 + 원본명)
        const timestamp = Date.now();
        const safeFilename = file.name.replace(/[^a-zA-Z0-9가-힣._-]/g, "_");
        const storedFilename = `${timestamp}_${safeFilename}`;
        const filepath = path.join(userUploadDir, storedFilename);

        // 파일 저장 (절대 경로 사용)
        await writeFile(filepath, fileBuffer);

        // 백엔드 /index API 호출
        // content 대신 file_path 전달 (백엔드에서 파싱)
        const response = await fetch(`${API_BASE_URL}/index`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                file_path: filepath,
                filename: file.name,
                user_id: userId,
            }),
        });

        let chunkCount = 0;
        if (response.ok) {
            const data = await response.json();
            chunkCount = data.chunk_count || 0;
        }

        // DB에 문서 정보 저장
        const document = await prisma.document.create({
            data: {
                filename: file.name,
                filepath: filepath,
                filesize: fileBuffer.length,
                mimetype: file.type || "text/plain",
                chunkCount: chunkCount,
                userId: userId,
            },
        });

        return NextResponse.json({
            message: `${file.name} 업로드 완료 (${chunkCount}개 청크 생성)`,
            chunk_count: chunkCount,
            filename: file.name,
            document_id: document.id,
        });
    } catch (error) {
        console.error("Upload error:", error);
        return NextResponse.json(
            { error: "서버 오류가 발생했습니다." },
            { status: 500 }
        );
    }
}

// 사용자의 문서 목록 조회
export async function GET(req: NextRequest) {
    try {
        const session = await auth();
        if (!session?.user?.id) {
            return NextResponse.json({ error: "인증이 필요합니다." }, { status: 401 });
        }

        const documents = await prisma.document.findMany({
            where: { userId: session.user.id },
            orderBy: { createdAt: "desc" },
            select: {
                id: true,
                filename: true,
                filesize: true,
                chunkCount: true,
                createdAt: true,
            },
        });

        return NextResponse.json({ documents });
    } catch (error) {
        console.error("Get documents error:", error);
        return NextResponse.json(
            { error: "서버 오류가 발생했습니다." },
            { status: 500 }
        );
    }
}
