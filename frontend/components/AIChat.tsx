"use client";

import { useEffect, useState } from "react";
import { HUDPanel } from "./HUDPanel";

interface AIChatProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading?: boolean;
  error?: string | null;
}

export function AIChat({
  value,
  onChange,
  onSubmit,
  loading,
  error,
}: AIChatProps) {
  const [openedAt, setOpenedAt] = useState("");

  useEffect(() => {
    const now = new Date();
    setOpenedAt(
      now.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    );
  }, []);

  return (
    <HUDPanel title="AI Assistant" subtitle="Argus search system online">
      <div className="space-y-3">
        <div className="rounded border border-cyber-cyan/30 bg-[#03070f] p-3 text-xs text-white/80 shadow-[0_0_18px_rgba(0,217,255,0.12)]">
          Earth monitoring system online. How can I assist you?
          <div className="mt-2 text-[10px] text-white/40">{openedAt}</div>
        </div>
        <div className="rounded border border-cyber-cyan/30 bg-[#03070f] px-3 py-2 shadow-[0_0_18px_rgba(0,217,255,0.12)]">
          <textarea
            className="w-full resize-none bg-transparent text-xs text-white placeholder:text-white/50 focus:outline-none"
            rows={3}
            placeholder="Type a message..."
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                onSubmit();
              }
            }}
          />
        </div>
        {error ? <div className="text-[10px] text-red-300">{error}</div> : null}
        <div className="flex justify-end">
          <button
            className="rounded border border-cyber-cyan/60 bg-cyber-cyan/20 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan hover:bg-cyber-cyan/30 disabled:cursor-not-allowed disabled:opacity-50"
            onClick={onSubmit}
            disabled={loading || value.trim().length === 0}
          >
            {loading ? "Searching..." : "Send"}
          </button>
        </div>
      </div>
    </HUDPanel>
  );
}
