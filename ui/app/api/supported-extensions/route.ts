import { NextResponse } from "next/server";

const API_BASE_URL = process.env.API_URL || process.env.API_BASE_URL || "http://127.0.0.1:8000";

// 백엔드 조회 실패 시 사용할 폴백 목록(클라이언트 UX 보호용).
const FALLBACK_EXTENSIONS = [".txt", ".md", ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".hwpx"];

// 업로드 가능한 확장자 목록을 백엔드(파서 레지스트리)에서 조회해 전달한다.
// 클라이언트 컴포넌트가 동일 출처로 호출할 수 있도록 하는 프록시이다.
export async function GET() {
    try {
        const res = await fetch(`${API_BASE_URL}/config/supported-extensions`, {
            headers: { "X-API-Key": process.env.API_KEY || "" },
            cache: "no-store",
        });
        if (!res.ok) throw new Error(`backend ${res.status}`);
        const data = await res.json();
        const extensions = Array.isArray(data.extensions) && data.extensions.length
            ? data.extensions
            : FALLBACK_EXTENSIONS;
        return NextResponse.json({ extensions });
    } catch {
        return NextResponse.json({ extensions: FALLBACK_EXTENSIONS });
    }
}
