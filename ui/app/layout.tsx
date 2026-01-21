import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { Providers } from "@/components/Providers";
import AppLayout from "@/components/layout/AppLayout";

const paperlogy = localFont({
  src: [
    { path: "../fonts/Paperlogy-1/Paperlogy-1Thin.ttf", weight: "100" },
    { path: "../fonts/Paperlogy-1/Paperlogy-2ExtraLight.ttf", weight: "200" },
    { path: "../fonts/Paperlogy-1/Paperlogy-3Light.ttf", weight: "300" },
    { path: "../fonts/Paperlogy-1/Paperlogy-4Regular.ttf", weight: "400" },
    { path: "../fonts/Paperlogy-1/Paperlogy-5Medium.ttf", weight: "500" },
    { path: "../fonts/Paperlogy-1/Paperlogy-6SemiBold.ttf", weight: "600" },
    { path: "../fonts/Paperlogy-1/Paperlogy-7Bold.ttf", weight: "700" },
    { path: "../fonts/Paperlogy-1/Paperlogy-8ExtraBold.ttf", weight: "800" },
    { path: "../fonts/Paperlogy-1/Paperlogy-9Black.ttf", weight: "900" },
  ],
  variable: "--font-paperlogy",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Terminal RAG",
  description: "RAG based Q&A System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <body className={`${paperlogy.variable} antialiased`}>
        <Providers>
          <AppLayout>{children}</AppLayout>
        </Providers>
      </body>
    </html>
  );
}
