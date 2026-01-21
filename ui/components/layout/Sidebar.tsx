"use client";

import { useSession, signOut } from "next-auth/react";
import { useState, useEffect } from "react";
import {
    MessageSquarePlus,
    Settings,
    LogOut,
    User,
    PanelLeftClose,
    PanelLeftOpen,
    Upload,
    FileText,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import SettingsDialog from "@/components/settings/SettingsDialog";
import { useChatStore } from "@/lib/chatStore";

export default function Sidebar() {
    const { data: session } = useSession();
    const router = useRouter();
    const chatStore = useChatStore();
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    // Get sorted sessions (newest first)
    const recentChats = mounted ? Object.values(chatStore.sessions).sort(
        (a, b) => b.updatedAt - a.updatedAt
    ) : [];

    const handleNewChat = () => {
        chatStore.createSession();
        router.push("/");
    };

    const handleSessionClick = (sessionId: string) => {
        chatStore.setCurrentSession(sessionId);
        router.push("/");
    };

    const handleDeleteSession = (e: React.MouseEvent, sessionId: string) => {
        e.stopPropagation();
        if (confirm("채팅 기록을 삭제하시겠습니까?")) {
            chatStore.deleteSession(sessionId);
        }
    };

    if (isCollapsed) {
        return (
            <div className="h-full w-16 bg-muted/10 border-r border-gray-200 dark:border-zinc-800 flex flex-col items-center py-4 z-40 transition-all duration-300 flex-shrink-0">
                <button
                    onClick={() => setIsCollapsed(false)}
                    className="p-2 text-muted-foreground hover:text-foreground hover:bg-accent/50 rounded-lg mb-4"
                >
                    <PanelLeftOpen size={20} />
                </button>
                <button
                    onClick={handleNewChat}
                    className="p-2 text-blue-500 hover:text-blue-600 hover:bg-blue-500/10 rounded-lg mb-2"
                >
                    <MessageSquarePlus size={24} />
                </button>
                <Link
                    href="/upload"
                    className="p-2 text-green-500 hover:text-green-600 hover:bg-green-500/10 rounded-lg mb-2"
                >
                    <Upload size={20} />
                </Link>
                <Link
                    href="/documents"
                    className="p-2 text-purple-500 hover:text-purple-600 hover:bg-purple-500/10 rounded-lg mb-4"
                >
                    <FileText size={20} />
                </Link>
                <div className="flex-1" />
                <button
                    onClick={() => setIsSettingsOpen(true)}
                    className="p-2 text-muted-foreground hover:text-foreground hover:bg-accent/50 rounded-lg mb-2"
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
            <div className="h-full w-64 bg-muted/10 border-r border-gray-200 dark:border-zinc-800 flex flex-col z-40 transition-all duration-300 flex-shrink-0">
                {/* Header */}
                <div className="p-4 flex items-center justify-between">
                    <Link href="/" className="flex items-center space-x-2">
                        <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                            <span className="text-white font-bold">R</span>
                        </div>
                        <span className="font-bold text-lg text-foreground">Terminal RAG</span>
                    </Link>
                    <button
                        onClick={() => setIsCollapsed(true)}
                        className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-accent/50 rounded-lg"
                    >
                        <PanelLeftClose size={18} />
                    </button>
                </div>

                {/* New Chat & Upload Buttons */}
                <div className="px-4 mb-4 space-y-2">
                    <button
                        onClick={handleNewChat}
                        className="w-full flex items-center justify-between px-4 py-3 bg-blue-500/10 hover:bg-blue-500/20 text-blue-500 hover:text-blue-600 rounded-xl transition border border-blue-500/20 group"
                    >
                        <span className="font-medium text-sm">새로운 채팅</span>
                        <MessageSquarePlus size={18} className="group-hover:scale-110 transition" />
                    </button>
                    <Link
                        href="/upload"
                        className="w-full flex items-center justify-between px-4 py-3 bg-green-500/10 hover:bg-green-500/20 text-green-500 hover:text-green-600 rounded-xl transition border border-green-500/20 group"
                    >
                        <span className="font-medium text-sm">문서 업로드</span>
                        <Upload size={18} className="group-hover:scale-110 transition" />
                    </Link>
                    <Link
                        href="/documents"
                        className="w-full flex items-center justify-between px-4 py-3 bg-purple-500/10 hover:bg-purple-500/20 text-purple-500 hover:text-purple-600 rounded-xl transition border border-purple-500/20 group"
                    >
                        <span className="font-medium text-sm">내 문서</span>
                        <FileText size={18} className="group-hover:scale-110 transition" />
                    </Link>
                </div>

                {/* Recent Chats (Scrollable) */}
                <div className="flex-1 overflow-y-auto px-2 space-y-1 custom-scrollbar">
                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
                        최근 대화
                    </div>
                    {recentChats.length === 0 ? (
                        <div className="px-4 py-8 text-center text-muted-foreground text-sm">
                            기록된 대화가 없습니다.
                        </div>
                    ) : (
                        recentChats.map((chat) => (
                            <div
                                key={chat.id}
                                onClick={() => handleSessionClick(chat.id)}
                                className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition group cursor-pointer ${chatStore.currentSessionId === chat.id
                                    ? "bg-accent text-accent-foreground"
                                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                                    }`}
                            >
                                <div className="flex items-center space-x-2 overflow-hidden">
                                    <MessageSquarePlus size={16} className="flex-shrink-0" />
                                    <span className="truncate">{chat.title}</span>
                                </div>
                                <button
                                    onClick={(e) => handleDeleteSession(e, chat.id)}
                                    className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition"
                                >
                                    <LogOut size={14} />
                                </button>
                            </div>
                        ))
                    )}
                </div>

                {/* Footer (User Profile) */}
                <div className="p-4 border-t border-gray-200 dark:border-zinc-800 bg-muted/20">
                    <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-3 overflow-hidden">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0 text-white font-bold text-xs ring-2 ring-border">
                                {session?.user?.name?.[0] || <User size={14} />}
                            </div>
                            <div className="flex-1 overflow-hidden">
                                <p className="text-sm font-medium text-foreground truncate">
                                    {session?.user?.name || "사용자"}
                                </p>
                                <p className="text-xs text-muted-foreground truncate">
                                    {session?.user?.email || "user@example.com"}
                                </p>
                            </div>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 mt-2">
                        <button
                            onClick={() => setIsSettingsOpen(true)}
                            className="flex items-center justify-center px-3 py-2 text-xs font-medium text-muted-foreground bg-background hover:bg-accent hover:text-foreground rounded-lg transition border border-gray-200 dark:border-zinc-800"
                        >
                            <Settings size={14} className="mr-1.5" />
                            설정
                        </button>
                        <button
                            onClick={() => signOut({ callbackUrl: "/login" })}
                            className="flex items-center justify-center px-3 py-2 text-xs font-medium text-red-500 bg-red-500/5 hover:bg-red-500/10 hover:text-red-600 rounded-lg transition border border-red-200 dark:border-red-900/20"
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
