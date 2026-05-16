import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { unlink } from "fs/promises";

const API_BASE_URL =
    process.env.API_URL || process.env.API_BASE_URL || "http://127.0.0.1:8000";

export async function DELETE(
    req: NextRequest,
    { params }: { params: Promise<{ id: string }> }
) {
    try {
        const session = await auth();
        if (!session?.user?.id) {
            return NextResponse.json({ error: "인증이 필요합니다." }, { status: 401 });
        }

        const { id } = await params;

        // 문서 조회 (본인 소유 확인)
        const document = await prisma.document.findFirst({
            where: {
                id: id,
                userId: session.user.id,
            },
        });

        if (!document) {
            return NextResponse.json({ error: "문서를 찾을 수 없습니다." }, { status: 404 });
        }

        // 파일 삭제 (존재하는 경우)
        try {
            await unlink(document.filepath);
        } catch (e) {
            // 파일이 이미 삭제된 경우 무시
            console.warn("File not found or already deleted:", document.filepath);
        }

        // 백엔드 인덱스(Qdrant + BM25) 청크 삭제
        // DB는 source-of-truth 이므로, 백엔드 호출 실패 시에도 DB 삭제는 진행한다.
        try {
            const resp = await fetch(`${API_BASE_URL}/index/by-source`, {
                method: "DELETE",
                headers: {
                    "Content-Type": "application/json",
                    "X-API-Key": process.env.API_KEY || "",
                },
                body: JSON.stringify({
                    filename: document.filename,
                    user_id: session.user.id,
                }),
                cache: "no-store",
            });
            if (!resp.ok) {
                const text = await resp.text();
                console.warn(
                    `백엔드 청크 삭제 실패 (status=${resp.status}): ${text}`
                );
            }
        } catch (e) {
            console.warn("백엔드 청크 삭제 호출 실패:", e);
        }

        // DB에서 삭제
        await prisma.document.delete({
            where: { id: id },
        });

        return NextResponse.json({ message: "문서가 삭제되었습니다." });
    } catch (error) {
        console.error("Delete document error:", error);
        return NextResponse.json(
            { error: "서버 오류가 발생했습니다." },
            { status: 500 }
        );
    }
}
