import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { getToken } from "next-auth/jwt";

export async function middleware(request: NextRequest) {
    const token = await getToken({
        req: request,
        secret: process.env.AUTH_SECRET,
    });

    const { pathname } = request.nextUrl;

    const isAuthPage =
        pathname.startsWith("/login") || pathname.startsWith("/register");

    const isApiAuthRoute = pathname.startsWith("/api/auth");
    const isPublicApiRoute =
        pathname.startsWith("/api/ask") ||
        pathname.startsWith("/api/search") ||
        pathname.startsWith("/api/health");

    // Allow auth API routes
    if (isApiAuthRoute) {
        return NextResponse.next();
    }

    // Allow public API routes
    if (isPublicApiRoute) {
        return NextResponse.next();
    }

    // Redirect to home if logged in user tries to access auth pages
    if (isAuthPage && token) {
        return NextResponse.redirect(new URL("/", request.url));
    }

    // Redirect to login if not authenticated
    if (!isAuthPage && !token) {
        return NextResponse.redirect(new URL("/login", request.url));
    }

    return NextResponse.next();
}

export const config = {
    matcher: ["/((?!_next/static|_next/image|favicon.ico|.*\\..*).*)"],
};
