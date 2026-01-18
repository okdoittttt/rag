"use client";

import { useSession, signOut } from "next-auth/react";
import { useState } from "react";
import {
    MessageSquarePlus,
    Settings,
    LogOut,
    User,
    PanelLeftClose,
    PanelLeftOpen,
    Upload,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import SettingsDialog from "@/components/settings/SettingsDialog";

export default function Sidebar() {
    const { data: session } = useSession();
    const router = useRouter();
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [isCollapsed, setIsCollapsed] = useState(false);

    // Mock chat history
    const recentChats: { id: string; title: string }[] = [
        // { id: "1", title: "RAG 시스템 아키텍처" },
        // { id: "2", title: "Docker Compose 설정" },
    ];

    const handleNewChat = () => {
        router.push("/");
        router.refresh();
    };

    if (isCollapsed) {
        return (
            <div className="h-full w-16 bg-[#18181b] border-r border-[#27272a] flex flex-col items-center py-4 z-40 transition-all duration-300 flex-shrink-0">
                <button
                    onClick={() => setIsCollapsed(false)}
                    className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg mb-4"
                >
                    <PanelLeftOpen size={20} />
                </button>
                <button
                    onClick={handleNewChat}
                    className="p-2 text-blue-400 hover:text-blue-300 hover:bg-blue-500/10 rounded-lg mb-2"
                >
                    <MessageSquarePlus size={24} />
                </button>
                <Link
                    href="/upload"
                    className="p-2 text-green-400 hover:text-green-300 hover:bg-green-500/10 rounded-lg mb-4"
                >
                    <Upload size={20} />
                </Link>
                <div className="flex-1" />
                <button
                    onClick={() => setIsSettingsOpen(true)}
                    className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg mb-2"
                >
                    <Settings size={20} />
                </button>
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-blue-500 flex items-center justify-center text-xs font-bold text-white mb-2 cursor-pointer" title={session?.user?.name || "사용자"}>
                    {session?.user?.name?.[0] || <User size={14} />}
                </div>

                <SettingsDialog
                    isOpen={isSettingsOpen}
                    onClose={() => setIsSettingsOpen(false)}
                />
            </div>
        );
    }

    return (
        <>
            <div className="h-full w-64 bg-[#18181b] border-r border-[#27272a] flex flex-col z-40 transition-all duration-300 flex-shrink-0">
                {/* Header */}
                <div className="p-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center space-x-2">
                        <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                            <span className="text-white font-bold">R</span>
                        </div>
                        <span className="font-bold text-lg text-white">Terminal RAG</span>
                    </Link>
                    <button
                        onClick={() => setIsCollapsed(true)}
                        className="p-1.5 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg"
                    >
                        <PanelLeftClose size={18} />
                    </button>
                </div>

                {/* New Chat & Upload Buttons */}
                <div className="px-4 mb-4 space-y-2">
                    <button
                        onClick={handleNewChat}
                        className="w-full flex items-center justify-between px-4 py-3 bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 hover:text-blue-300 rounded-xl transition border border-blue-500/20 group"
                    >
                        <span className="font-medium text-sm">새로운 채팅</span>
                        <MessageSquarePlus size={18} className="group-hover:scale-110 transition" />
                    </button>
                    <Link
                        href="/upload"
                        className="w-full flex items-center justify-between px-4 py-3 bg-green-600/10 hover:bg-green-600/20 text-green-400 hover:text-green-300 rounded-xl transition border border-green-500/20 group"
                    >
                        <span className="font-medium text-sm">문서 업로드</span>
                        <Upload size={18} className="group-hover:scale-110 transition" />
                    </Link>
                </div>

                {/* Recent Chats (Scrollable) */}
                <div className="flex-1 overflow-y-auto px-2 space-y-1 custom-scrollbar">
                    <div className="px-2 py-1.5 text-xs font-semibold text-gray-500">
                        최근 대화
                    </div>
                    {recentChats.length === 0 ? (
                        <div className="px-4 py-8 text-center text-gray-500 text-sm">
                            기록된 대화가 없습니다.
                        </div>
                    ) : (
                        recentChats.map((chat) => (
                            <button
                                key={chat.id}
                                className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-white/5 hover:text-white rounded-lg transition truncate"
                            >
                                {chat.title}
                            </button>
                        ))
                    )}
                </div>

                {/* Footer (User Profile) */}
                <div className="p-4 border-t border-[#27272a] bg-[#1d1d20]">
                    <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-3 overflow-hidden">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0 text-white font-bold text-xs ring-2 ring-[#27272a]">
                                {session?.user?.name?.[0] || <User size={14} />}
                            </div>
                            <div className="flex-1 overflow-hidden">
                                <p className="text-sm font-medium text-white truncate">
                                    {session?.user?.name || "사용자"}
                                </p>
                                <p className="text-xs text-gray-500 truncate">
                                    {session?.user?.email || "user@example.com"}
                                </p>
                            </div>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 mt-2">
                        <button
                            onClick={() => setIsSettingsOpen(true)}
                            className="flex items-center justify-center px-3 py-2 text-xs font-medium text-gray-400 bg-white/5 hover:bg-white/10 hover:text-white rounded-lg transition"
                        >
                            <Settings size={14} className="mr-1.5" />
                            설정
                        </button>
                        <button
                            onClick={() => signOut({ callbackUrl: "/login" })}
                            className="flex items-center justify-center px-3 py-2 text-xs font-medium text-red-400 bg-red-500/5 hover:bg-red-500/10 hover:text-red-300 rounded-lg transition"
                        >
                            <LogOut size={14} className="mr-1.5" />
                            로그아웃
                        </button>
                    </div>
                </div>
            </div>

            <SettingsDialog
                isOpen={isSettingsOpen}
                onClose={() => setIsSettingsOpen(false)}
            />
        </>
    );
}
