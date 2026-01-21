import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();

        // 백엔드 API (localhost:8000) 호출
        // Next.js 서버는 호스트 머신에서 실행되므로 localhost:8000으로 접근 가능
        const response = await fetch("http://127.0.0.1:8000/ask/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            return NextResponse.json(
                { message: `Backend error: ${response.status}` },
                { status: response.status }
            );
        }

        // 스트림을 그대로 파이핑
        return new Response(response.body, {
            headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        });
    } catch (error) {
        console.error("Proxy error:", error);
        return NextResponse.json(
            { message: "Internal Proxy Error" },
            { status: 500 }
        );
    }
}
