import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

interface SettingsState {
    geminiApiKey: string;
    geminiModel: string;
    ollamaBaseUrl: string;
    ollamaModel: string;
    setGeminiApiKey: (key: string) => void;
    setGeminiModel: (model: string) => void;
    setOllamaBaseUrl: (url: string) => void;
    setOllamaModel: (model: string) => void;
}

export const useSettingsStore = create<SettingsState>()(
    persist(
        (set) => ({
            geminiApiKey: '',
            geminiModel: 'gemini-pro',
            ollamaBaseUrl: 'http://localhost:11434',
            ollamaModel: 'llama3',
            setGeminiApiKey: (key) => set({ geminiApiKey: key }),
            setGeminiModel: (model) => set({ geminiModel: model }),
            setOllamaBaseUrl: (url) => set({ ollamaBaseUrl: url }),
            setOllamaModel: (model) => set({ ollamaModel: model }),
        }),
        {
            name: 'rag-settings', // name of the item in the storage (must be unique)
            storage: createJSONStorage(() => localStorage),
        }
    )
);
