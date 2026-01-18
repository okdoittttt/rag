"use client";

import { useEffect, useState } from "react";
import { useSettingsStore } from "@/lib/store";
import { X, Save } from "lucide-react";

interface SettingsDialogProps {
    isOpen: boolean;
    onClose: () => void;
}

export default function SettingsDialog({ isOpen, onClose }: SettingsDialogProps) {
    const settings = useSettingsStore();

    // Local state for form inputs to avoid direct store updates on every keystroke
    // or just use store directly since it's local only? specific save button requested.
    // "저장 버튼" requested. So local state -> sync on save.
    const [geminiApiKey, setGeminiApiKey] = useState(settings.geminiApiKey);
    const [geminiModel, setGeminiModel] = useState(settings.geminiModel);
    const [ollamaBaseUrl, setOllamaBaseUrl] = useState(settings.ollamaBaseUrl);
    const [ollamaModel, setOllamaModel] = useState(settings.ollamaModel);
    const [activeTab, setActiveTab] = useState("gemini");

    // Sync state when dialog opens
    useEffect(() => {
        if (isOpen) {
            setGeminiApiKey(settings.geminiApiKey);
            setGeminiModel(settings.geminiModel);
            setOllamaBaseUrl(settings.ollamaBaseUrl);
            setOllamaModel(settings.ollamaModel);
        }
    }, [isOpen, settings]);

    const handleSave = () => {
        settings.setGeminiApiKey(geminiApiKey);
        settings.setGeminiModel(geminiModel);
        settings.setOllamaBaseUrl(ollamaBaseUrl);
        settings.setOllamaModel(ollamaModel);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-[#1e1e1e] w-full max-w-lg rounded-xl border border-white/10 shadow-2xl overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-[#252525]">
                    <h2 className="text-lg font-semibold text-white">설정</h2>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-white transition"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6">
                    {/* Tabs */}
                    <div className="flex space-x-4 mb-6 border-b border-white/10">
                        <button
                            onClick={() => setActiveTab("gemini")}
                            className={`pb-2 px-1 text-sm font-medium transition relative ${activeTab === "gemini"
                                    ? "text-blue-400"
                                    : "text-gray-400 hover:text-white"
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
                                    ? "text-blue-400"
                                    : "text-gray-400 hover:text-white"
                                }`}
                        >
                            Ollama
                            {activeTab === "ollama" && (
                                <div className="absolute bottom-0 left-0 w-full h-0.5 bg-blue-500 rounded-t-full" />
                            )}
                        </button>
                    </div>

                    {/* Tab Panels */}
                    <div className="space-y-4">
                        {activeTab === "gemini" && (
                            <div className="space-y-4 animate-in fade-in zoom-in-95 duration-200">
                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-1">
                                        API Key
                                    </label>
                                    <input
                                        type="password"
                                        value={geminiApiKey}
                                        onChange={(e) => setGeminiApiKey(e.target.value)}
                                        className="w-full px-4 py-2.5 bg-black/20 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-gray-600"
                                        placeholder="AIzaSy..."
                                    />
                                    <p className="text-xs text-gray-500 mt-1">
                                        Google AI Studio에서 발급받은 API 키를 입력하세요.
                                    </p>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-1">
                                        Model Name
                                    </label>
                                    <input
                                        type="text"
                                        value={geminiModel}
                                        onChange={(e) => setGeminiModel(e.target.value)}
                                        className="w-full px-4 py-2.5 bg-black/20 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-gray-600"
                                        placeholder="gemini-pro"
                                    />
                                </div>
                            </div>
                        )}

                        {activeTab === "ollama" && (
                            <div className="space-y-4 animate-in fade-in zoom-in-95 duration-200">
                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-1">
                                        Base URL
                                    </label>
                                    <input
                                        type="text"
                                        value={ollamaBaseUrl}
                                        onChange={(e) => setOllamaBaseUrl(e.target.value)}
                                        className="w-full px-4 py-2.5 bg-black/20 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-gray-600"
                                        placeholder="http://localhost:11434"
                                    />
                                    <p className="text-xs text-gray-500 mt-1">
                                        Ollama 서버의 주소입니다. (예: http://localhost:11434)
                                    </p>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-1">
                                        Model Name
                                    </label>
                                    <input
                                        type="text"
                                        value={ollamaModel}
                                        onChange={(e) => setOllamaModel(e.target.value)}
                                        className="w-full px-4 py-2.5 bg-black/20 border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition placeholder-gray-600"
                                        placeholder="llama3"
                                    />
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end px-6 py-4 border-t border-white/10 bg-[#252525]">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-white transition mr-2"
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
