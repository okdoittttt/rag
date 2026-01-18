import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { unlink } from "fs/promises";

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

        // DB에서 삭제
        await prisma.document.delete({
            where: { id: id },
        });

        // TODO: Qdrant에서 해당 문서의 청크도 삭제 (user_id + filename 필터)
        // 현재는 청크는 Qdrant에 남아있음 (추후 구현 필요)

        return NextResponse.json({ message: "문서가 삭제되었습니다." });
    } catch (error) {
        console.error("Delete document error:", error);
        return NextResponse.json(
            { error: "서버 오류가 발생했습니다." },
            { status: 500 }
        );
    }
}
