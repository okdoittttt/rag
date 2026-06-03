import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    // proxy(미들웨어)가 버퍼링하는 요청 본문의 최대 크기. 기본 10MB.
    // 대용량 업로드 대비 안전망으로 상향한다. (16.1.3 기준 정식 옵션,
    // 구 middlewareClientMaxBodySize 는 deprecated)
    proxyClientMaxBodySize: "100mb",
  },
  async rewrites() {
    const backendUrl = process.env.API_URL || "http://127.0.0.1:8000";
    return [
      // Swagger 문서만 백엔드로 프록시 (API 엔드포인트는 Next.js API Route에서 인증 후 프록시)
      {
        source: "/docs",
        destination: `${backendUrl}/docs`,
      },
      {
        source: "/openapi.json",
        destination: `${backendUrl}/openapi.json`,
      },
    ];
  },
};

export default nextConfig;
