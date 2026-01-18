import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/:path*", // Backend API
      },
      {
        source: "/docs",
        destination: "http://127.0.0.1:8000/docs",
      },
      {
        source: "/openapi.json",
        destination: "http://127.0.0.1:8000/openapi.json",
      },
    ];
  },
};

export default nextConfig;
