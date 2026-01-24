"use client";

import { useEffect, useState } from "react";
import { useSettingsStore } from "@/lib/store";
import { X, Save } from "lucide-react";
import { useTheme } from "next-themes";
import { getSystemPrompt, updateSystemPrompt } from "@/lib/api";

interface SettingsDialogProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function SettingsDialog({ isOpen, onClose }: SettingsDialogProps) {
    const settings = useSettingsStore();
    const { theme, setTheme, resolvedTheme } = useTheme();

    // Local state for form inputs
    const [geminiApiKey, setGeminiApiKey] = useState(settings.geminiApiKey);
    const [geminiModel, setGeminiModel] = useState(settings.geminiModel);
    const [ollamaBaseUrl, setOllamaBaseUrl] = useState(settings.ollamaBaseUrl);
    const [ollamaModel, setOllamaModel] = useState(settings.ollamaModel);

    // New state for System Prompt
    const [systemPrompt, setSystemPrompt] = useState("");
    const [isLoadingPrompt, setIsLoadingPrompt] = useState(false);

    const [activeTab, setActiveTab] = useState("general"); // Default to general

    // Sync state when dialog opens
    useEffect(() => {
        if (isOpen) {
            setGeminiApiKey(settings.geminiApiKey);
            setGeminiModel(settings.geminiModel);
            setOllamaBaseUrl(settings.ollamaBaseUrl);
            setOllamaModel(settings.ollamaModel);

            // Load system prompt
            setIsLoadingPrompt(true);
            getSystemPrompt()
                .then(setSystemPrompt)
                .catch((err) => {
                    console.error("Failed to load system prompt:", err);
                    setSystemPrompt("Failed to load prompt. Please try again.");
                })
                .finally(() => setIsLoadingPrompt(false));
        }
    }, [isOpen, settings]);

    const handleSave = async () => {
        try {
            // Save System Prompt first (async)
            await updateSystemPrompt(systemPrompt);

            // Save other settings (sync/store)
            settings.setGeminiApiKey(geminiApiKey);
            settings.setGeminiModel(geminiModel);
            settings.setOllamaBaseUrl(ollamaBaseUrl);
            settings.setOllamaModel(ollamaModel);

            onClose();
        } catch (error) {
            console.error("Failed to save settings:", error);
            alert("설정 저장에 실패했습니다. (시스템 프롬프트 업데이트 실패)");
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-background w-full max-w-lg rounded-xl border border-border shadow-2xl overflow-hidden transition-colors">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-muted/30">
                    <h2 className="text-lg font-semibold text-foreground">설정</h2>
                    <button
                        onClick={onClose}
                        className="text-muted-foreground hover:text-foreground transition"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6">
                    {/* Tabs */}
                    <div className="flex space-x-4 mb-6 border-b border-border">
                        <button
                            onClick={() => setActiveTab("general")}
                            className={`pb-2 px-1 text-sm font-medium transition relative ${activeTab === "general"
                                ? "text-blue-500 font-bold"
                                : "text-muted-foreground hover:text-foreground"
                                }`}
                        >
                            기본 설정
                            {activeTab === "general" && (
                                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-500 rounded-t-full" />
                            )}
                        </button>
                        <button
                            onClick={() => setActiveTab("gemini")}
                            className={`pb-2 px-1 text-sm font-medium transition relative ${activeTab === "gemini"
                                ? "text-blue-500 font-bold"
                                : "text-muted-foreground hover:text-foreground"
                                }`}
                        >
                            Gemini
                            {activeTab === "gemini" && (
                                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-500 rounded-t-full" />
                            )}
                        </button>
                        <button
                            onClick={() => setActiveTab("ollama")}
                            className={`pb-2 px-1 text-sm font-medium transition relative ${activeTab === "ollama"
                                ? "text-blue-500 font-bold"
                                : "text-muted-foreground hover:text-foreground"
                                }`}
                        >
                            Ollama
                            {activeTab === "ollama" && (
                                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-500 rounded-t-full" />
                            )}
                        </button>
                        <button
                            onClick={() => setActiveTab("prompt")}
                            className={`pb-2 px-1 text-sm font-medium transition relative ${activeTab === "prompt"
                                ? "text-blue-500 font-bold"
                                : "text-muted-foreground hover:text-foreground"
                                }`}
                        >
                            프롬프트
                            {activeTab === "prompt" && (
                                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-500 rounded-t-full" />
                            )}
                        </button>
                    </div>

                    {/* Tab Panels */}
                    <div className="space-y-4 min-h-[200px]">
                        {/* General Tab */}
                        {activeTab === "general" && (
                            <div className="space-y-4 animate-in fade-in zoom-in-95 duration-200">
                                <div className="flex items-center justify-between">
                                    <div className="space-y-0.5">
                                        <label className="block text-sm font-medium text-foreground">
                                            다크 모드
                                        </label>
                                        <p className="text-xs text-muted-foreground">
                                            화면을 어둡게 표시하여 눈의 피로를 줄입니다.
                                        </p>
                                    </div>
                                    {(() => {
                                        const isDark = theme === 'dark' || (theme === 'system' && resolvedTheme === 'dark');
                                        return (
                                            <button
                                                type="button"
                                                role="switch"
                                                aria-checked={isDark}
                                                aria-label="다크 모드 설정"
                                                onClick={() => setTheme(isDark ? 'light' : 'dark')}
                                                className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 ${isDark ? 'bg-green-500' : 'bg-gray-200 dark:bg-gray-700'
                                                    }`}
                                            >
                                                <span
                                                    aria-hidden="true"
                                                    className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-md ring-0 transition duration-200 ease-in-out ${isDark ? 'translate-x-5' : 'translate-x-0'
                                                        }`}
                                                />
                                            </button>
                                        );
                                    })()}
                                </div>
                                <div className="mt-2 text-right">
                                    {theme !== 'system' && (
                                        <button
                                            onClick={() => setTheme('system')}
                                            className="text-xs text-blue-500 hover:text-blue-600 underline"
                                        >
                                            시스템 설정 따르기
                                        </button>
                                    )}
                                </div>
                            </div>
                        )}

                        {activeTab === "gemini" && (
                            <div className="space-y-4 animate-in fade-in zoom-in-95 duration-200">
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">
                                        API Key
                                    </label>
                                    <input
                                        type="password"
                                        value={geminiApiKey}
                                        onChange={(e) => setGeminiApiKey(e.target.value)}
                                        className="w-full px-4 py-2.5 bg-muted/50 border border-border rounded-lg text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-muted-foreground"
                                        placeholder="AIzaSy..."
                                    />
                                    <p className="text-xs text-muted-foreground mt-1">
                                        Google AI Studio에서 발급받은 API 키를 입력하세요.
                                    </p>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">
                                        Model Name
                                    </label>
                                    <input
                                        type="text"
                                        value={geminiModel}
                                        onChange={(e) => setGeminiModel(e.target.value)}
                                        className="w-full px-4 py-2.5 bg-muted/50 border border-border rounded-lg text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-muted-foreground"
                                        placeholder="gemini-pro"
                                    />
                                </div>
                            </div>
                        )}

                        {activeTab === "ollama" && (
                            <div className="space-y-4 animate-in fade-in zoom-in-95 duration-200">
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">
                                        Base URL
                                    </label>
                                    <input
                                        type="text"
                                        value={ollamaBaseUrl}
                                        onChange={(e) => setOllamaBaseUrl(e.target.value)}
                                        className="w-full px-4 py-2.5 bg-muted/50 border border-border rounded-lg text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-muted-foreground"
                                        placeholder="http://localhost:11434"
                                    />
                                    <p className="text-xs text-muted-foreground mt-1">
                                        Ollama 서버의 주소입니다. (예: http://localhost:11434)
                                    </p>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">
                                        Model Name
                                    </label>
                                    <input
                                        type="text"
                                        value={ollamaModel}
                                        onChange={(e) => setOllamaModel(e.target.value)}
                                        className="w-full px-4 py-2.5 bg-muted/50 border border-border rounded-lg text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-muted-foreground"
                                        placeholder="llama3"
                                    />
                                </div>
                            </div>
                        )}

                        {activeTab === "prompt" && (
                            <div className="space-y-4 animate-in fade-in zoom-in-95 duration-200">
                                <div>
                                    <label className="block text-sm font-medium text-foreground mb-1">
                                        System Prompt
                                    </label>
                                    <div className="relative">
                                        <textarea
                                            value={systemPrompt}
                                            onChange={(e) => setSystemPrompt(e.target.value)}
                                            disabled={isLoadingPrompt}
                                            className="w-full h-96 px-4 py-3 bg-muted/50 border border-border rounded-lg text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-muted-foreground resize-none font-mono leading-relaxed"
                                            placeholder="시스템 프롬프트를 입력하세요..."
                                        />
                                        {isLoadingPrompt && (
                                            <div className="absolute inset-0 flex items-center justify-center bg-black/10 rounded-lg">
                                                <div className="animate-spin h-5 w-5 border-2 border-blue-500 rounded-full border-t-transparent"></div>
                                            </div>
                                        )}
                                    </div>
                                    <p className="text-xs text-muted-foreground mt-2">
                                        AI 어시스턴트의 페르소나와 제약조건을 설정합니다. 변경사항은 서버에 텍스트 파일로 저장됩니다.
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end px-6 py-4 border-t border-border bg-muted/30">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition mr-2"
                    >
                        취소
                    </button>
                    <button
                        onClick={handleSave}
                        className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition shadow-lg shadow-blue-900/20"
                    >
                        <Save size={16} className="mr-2" />
                        저장
                    </button>
                </div>
            </div>
        </div>
    );
}
