import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { Message } from '@/components/chat';

export interface ChatSession {
    id: string;
    title: string;
    messages: Message[];
    updatedAt: number;
}

interface ChatStore {
    sessions: Record<string, ChatSession>;
    currentSessionId: string | null;

    // Actions
    createSession: () => string;
    setCurrentSession: (id: string) => void;
    addMessage: (sessionId: string, message: Message) => void;
    updateMessage: (sessionId: string, messageId: string, updates: Partial<Message>) => void;
    appendMessageContent: (sessionId: string, messageId: string, content: string) => void;
    updateSessionTitle: (sessionId: string, title: string) => void;
    deleteSession: (sessionId: string) => void;
    clearSessions: () => void;

    // Helper
    getCurrentSession: () => ChatSession | undefined;
}

export const useChatStore = create<ChatStore>()(
    persist(
        (set, get) => ({
            sessions: {},
            currentSessionId: null,

            createSession: () => {
                const id = Date.now().toString();
                const newSession: ChatSession = {
                    id,
                    title: "새로운 채팅",
                    messages: [],
                    updatedAt: Date.now(),
                };
                set((state) => ({
                    sessions: { ...state.sessions, [id]: newSession },
                    currentSessionId: id,
                }));
                return id;
            },

            setCurrentSession: (id) => {
                set({ currentSessionId: id });
            },

            addMessage: (sessionId, message) => {
                set((state) => {
                    const session = state.sessions[sessionId];
                    if (!session) return state;

                    const updatedSession = {
                        ...session,
                        messages: [...session.messages, message],
                        updatedAt: Date.now(),
                    };

                    // Auto-generate title from first user message if it's "New Chat"
                    if (session.title === "새로운 채팅" && message.role === "user") {
                        updatedSession.title = message.content.slice(0, 30) + (message.content.length > 30 ? "..." : "");
                    }

                    return {
                        sessions: { ...state.sessions, [sessionId]: updatedSession },
                    };
                });
            },

            updateMessage: (sessionId, messageId, updates) => {
                set((state) => {
                    const session = state.sessions[sessionId];
                    if (!session) return state;

                    const updatedMessages = session.messages.map((msg) =>
                        msg.id === messageId ? { ...msg, ...updates } : msg
                    );

                    return {
                        sessions: {
                            ...state.sessions,
                            [sessionId]: { ...session, messages: updatedMessages, updatedAt: Date.now() },
                        },
                    };
                });
            },

            appendMessageContent: (sessionId, messageId, content) => {
                set((state) => {
                    const session = state.sessions[sessionId];
                    if (!session) return state;

                    const updatedMessages = session.messages.map((msg) =>
                        msg.id === messageId ? { ...msg, content: msg.content + content } : msg
                    );

                    return {
                        sessions: {
                            ...state.sessions,
                            [sessionId]: { ...session, messages: updatedMessages, updatedAt: Date.now() },
                        },
                    };
                });
            },

            updateSessionTitle: (sessionId, title) => {
                set((state) => {
                    const session = state.sessions[sessionId];
                    if (!session) return state;
                    return {
                        sessions: {
                            ...state.sessions,
                            [sessionId]: { ...session, title, updatedAt: Date.now() },
                        },
                    };
                });
            },

            deleteSession: (sessionId) => {
                set((state) => {
                    const newSessions = { ...state.sessions };
                    delete newSessions[sessionId];

                    let nextSessionId = state.currentSessionId;
                    if (state.currentSessionId === sessionId) {
                        nextSessionId = null; // Or select another one?
                    }

                    return {
                        sessions: newSessions,
                        currentSessionId: nextSessionId,
                    };
                });
            },

            clearSessions: () => set({ sessions: {}, currentSessionId: null }),

            getCurrentSession: () => {
                const state = get();
                if (!state.currentSessionId) return undefined;
                return state.sessions[state.currentSessionId];
            },
        }),
        {
            name: 'rag-chat-history',
            storage: createJSONStorage(() => localStorage),
        }
    )
);
