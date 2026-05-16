import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";

const API_BASE_URL = process.env.API_URL || "http://127.0.0.1:8000";

export async function GET() {
    try {
        const session = await auth();
        if (!session?.user?.id) {
            return NextResponse.json({ error: "인증이 필요합니다." }, { status: 401 });
        }

        const response = await fetch(`${API_BASE_URL}/config/system-prompt`, {
            method: "GET",
            headers: {
                "X-API-Key": process.env.API_KEY || "",
                "Connection": "close",
            },
            cache: "no-store",
        });

        if (!response.ok) {
            const errorText = await response.text();
            return NextResponse.json(
                { error: `Backend error: ${response.status} - ${errorText}` },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error("Failed to fetch system prompt:", error);
        return NextResponse.json(
            { error: "Failed to fetch system prompt from backend" },
            { status: 500 }
        );
    }
}

export async function POST(req: NextRequest) {
    try {
        const session = await auth();
        if (!session?.user?.id) {
            return NextResponse.json({ error: "인증이 필요합니다." }, { status: 401 });
        }

        const body = await req.json();

        const response = await fetch(`${API_BASE_URL}/config/system-prompt`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-API-Key": process.env.API_KEY || "",
                "Connection": "close",
            },
            body: JSON.stringify(body),
            cache: "no-store",
        });

        if (!response.ok) {
            const errorText = await response.text();
            return NextResponse.json(
                { error: `Backend error: ${response.status} - ${errorText}` },
                { status: response.status }
            );
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error("Failed to update system prompt:", error);
        return NextResponse.json(
            { error: "Failed to update system prompt" },
            { status: 500 }
        );
    }
}
