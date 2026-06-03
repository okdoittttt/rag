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

// 백엔드(파서 레지스트리) 조회 실패 시 사용할 폴백 목록.
const FALLBACK_EXTENSIONS = [".txt", ".md", ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".hwpx"];

/** 업로드 허용 확장자를 백엔드에서 조회한다(실패 시 폴백). */
async function getAllowedExtensions(): Promise<string[]> {
    try {
        const res = await fetch(`${API_BASE_URL}/config/supported-extensions`, {
            headers: { "X-API-Key": process.env.API_KEY || "" },
            cache: "no-store",
        });
        if (!res.ok) throw new Error(`backend ${res.status}`);
        const data = await res.json();
        return Array.isArray(data.extensions) && data.extensions.length
            ? data.extensions
            : FALLBACK_EXTENSIONS;
    } catch {
        return FALLBACK_EXTENSIONS;
    }
}

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

        // 파일 확장자 검증 (허용 목록은 백엔드 파서 레지스트리에서 조회)
        const allowedExtensions = await getAllowedExtensions();
        const ext = "." + originalFilename.split(".").pop()?.toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            return NextResponse.json(
                { error: `지원되지 않는 파일 형식입니다. (${allowedExtensions.join(", ")}만 지원)` },
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
        const mimetype = req.headers.get("content-type") || "application/octet-stream";

        // 백엔드 /index/stream(SSE) 호출 → 진행 이벤트를 클라이언트로 그대로 relay
        const backendRes = await fetch(`${API_BASE_URL}/index/stream`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-API-Key": process.env.API_KEY || "",
            },
            body: JSON.stringify({
                file_path: filepath,
                filename: originalFilename,
                user_id: userId,
            }),
            cache: "no-store",
        });

        if (!backendRes.ok || !backendRes.body) {
            const errorText = await backendRes.text().catch(() => "");
            console.error(`Backend stream error ${backendRes.status}:`, errorText);
            return NextResponse.json(
                { error: `백엔드 처리 실패: ${backendRes.status}` },
                { status: backendRes.status || 502 }
            );
        }

        // SSE를 클라이언트로 파이핑하면서 done 이벤트의 chunk_count를 가로채,
        // 스트림 종료(flush) 시점에 문서 메타데이터를 DB에 저장한다.
        const decoder = new TextDecoder();
        let buffer = "";
        let chunkCount = 0;
        let sawError = false;

        const transform = new TransformStream({
            transform(chunk, controller) {
                controller.enqueue(chunk); // 클라이언트로 그대로 relay
                buffer += decoder.decode(chunk, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";
                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const data = line.slice(6);
                    if (data === "[DONE]") continue;
                    try {
                        const ev = JSON.parse(data);
                        if (ev.phase === "done") {
                            chunkCount = ev.chunk_count ?? 0;
                        } else if (ev.phase === "error") {
                            sawError = true;
                        }
                    } catch {
                        // 진행 이벤트 파싱 실패는 무시 (relay에는 영향 없음)
                    }
                }
            },
            async flush() {
                if (sawError) return;
                try {
                    await prisma.document.create({
                        data: {
                            filename: originalFilename,
                            filepath: filepath,
                            filesize: filesize,
                            mimetype: mimetype,
                            chunkCount: chunkCount,
                            userId: userId,
                        },
                    });
                } catch (e) {
                    console.error("DB save error after indexing:", e);
                }
            },
        });

        return new Response(backendRes.body.pipeThrough(transform), {
            headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
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
