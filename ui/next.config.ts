import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path((?!auth|upload|documents).*)",
        destination: `${process.env.API_URL || "http://127.0.0.1:8000"}/:path*`, // Backend API (exclude Next.js API routes)
      },
      {
        source: "/docs",
        destination: `${process.env.API_URL || "http://127.0.0.1:8000"}/docs`,
      },
      {
        source: "/openapi.json",
        destination: `${process.env.API_URL || "http://127.0.0.1:8000"}/openapi.json`,
      },
    ];
  },
};

export default nextConfig;
