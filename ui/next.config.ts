import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
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
