import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { createWriteStream } from "fs";
import { mkdir, stat } from "fs/promises";
import { Readable } from "stream";
import { pipeline } from "stream/promises";
import path from "path";

const API_BASE_URL = process.env.API_URL || process.env.API_BASE_URL || "http://127.0.0.1:8000";
const UPLOAD_DIR = process.env.UPLOAD_DIR || "./data/uploads";
const ALLOWED_EXTENSIONS = [".txt", ".md", ".pdf"];

export async function POST(req: NextRequest) {
    try {
        // 세션에서 user_id 추출 (이 라우트는 미들웨어 matcher에서 제외되므로
        // 인증을 핸들러에서 직접 수행한다)
        const session = await auth();
        if (!session?.user?.id) {
            return NextResponse.json({ error: "인증이 필요합니다." }, { status: 401 });
        }

        const userId = session.user.id;

        // 파일명은 x-filename 헤더로 전달받는다 (본문은 파일 raw 바이트 스트림).
        const rawFilename = req.headers.get("x-filename");
        if (!rawFilename) {
            return NextResponse.json(
                { error: "파일명이 없습니다. (x-filename 헤더 필요)" },
                { status: 400 }
            );
        }
        const originalFilename = decodeURIComponent(rawFilename);

        // 파일 확장자 검증
        const ext = "." + originalFilename.split(".").pop()?.toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            return NextResponse.json(
                { error: "지원되지 않는 파일 형식입니다. (.txt, .md, .pdf만 지원)" },
                { status: 400 }
            );
        }

        if (!req.body) {
            return NextResponse.json(
                { error: "업로드할 파일 본문이 없습니다." },
                { status: 400 }
            );
        }

        // 사용자별 업로드 디렉토리 생성
        const userUploadDir = path.resolve(UPLOAD_DIR, userId);
        await mkdir(userUploadDir, { recursive: true });

        // 고유 파일명 생성 (타임스탬프 + 원본명 sanitize)
        const timestamp = Date.now();
        const safeFilename = originalFilename.replace(/[^a-zA-Z0-9가-힣._-]/g, "_");
        const storedFilename = `${timestamp}_${safeFilename}`;
        const filepath = path.join(userUploadDir, storedFilename);

        // 파일을 메모리에 통째로 적재하지 않고 스트리밍으로 디스크에 기록한다.
        const nodeStream = Readable.fromWeb(req.body as Parameters<typeof Readable.fromWeb>[0]);
        await pipeline(nodeStream, createWriteStream(filepath));

        // 저장된 파일 크기 (스트리밍이므로 디스크에서 조회)
        const { size: filesize } = await stat(filepath);

        // 백엔드 /index API 호출 (content 대신 file_path 전달, 백엔드에서 파싱)
        console.log(`Sending upload request to backend: ${API_BASE_URL}/index`);

        let response;
        try {
            response = await fetch(`${API_BASE_URL}/index`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": process.env.API_KEY || "",
                    "Connection": "close",
                },
                body: JSON.stringify({
                    file_path: filepath,
                    filename: originalFilename,
                    user_id: userId,
                }),
                cache: "no-store",
            });
        } catch (fetchError) {
            console.error(`Fetch failed for ${API_BASE_URL}/index:`, fetchError);
            throw fetchError;
        }

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`Backend returned error ${response.status}:`, errorText);
            return NextResponse.json(
                { error: `백엔드 처리 실패: ${response.status} - ${errorText}` },
                { status: response.status }
            );
        }

        const data = await response.json();
        const chunkCount = data.chunk_count || 0;

        // DB에 문서 정보 저장
        const document = await prisma.document.create({
            data: {
                filename: originalFilename,
                filepath: filepath,
                filesize: filesize,
                mimetype: req.headers.get("content-type") || "application/octet-stream",
                chunkCount: chunkCount,
                userId: userId,
            },
        });

        return NextResponse.json({
            message: `${originalFilename} 업로드 완료 (${chunkCount}개 청크 생성)`,
            chunk_count: chunkCount,
            filename: originalFilename,
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
