"use client";

import { Check, ChevronsUpDown } from "lucide-react";

interface ModelSelectorProps {
    value: "gemini" | "ollama";
    onChange: (value: "gemini" | "ollama") => void;
    disabled?: boolean;
}

export function ModelSelector({ value, onChange, disabled }: ModelSelectorProps) {
    return (
        <div className="relative inline-block text-left">
            <div className="flex items-center space-x-2 bg-muted/50 rounded-lg p-1 border">
                <button
                    onClick={() => onChange("gemini")}
                    disabled={disabled}
                    className={`px-3 py-1.5 text-sm rounded-md transition-all flex items-center gap-2 ${value === "gemini"
                            ? "bg-background shadow-sm text-foreground font-medium"
                            : "text-muted-foreground hover:bg-muted"
                        }`}
                >
                    <span>Gemini</span>
                    {value === "gemini" && <Check size={14} className="text-blue-500" />}
                </button>
                <button
                    onClick={() => onChange("ollama")}
                    disabled={disabled}
                    className={`px-3 py-1.5 text-sm rounded-md transition-all flex items-center gap-2 ${value === "ollama"
                            ? "bg-background shadow-sm text-foreground font-medium"
                            : "text-muted-foreground hover:bg-muted"
                        }`}
                >
                    <span>Ollama</span>
                    {value === "ollama" && <Check size={14} className="text-blue-500" />}
                </button>
            </div>
        </div>
    );
}
