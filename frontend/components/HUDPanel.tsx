"use client";

import type { ReactNode } from "react";

interface HUDPanelProps {
  title: string;
  subtitle?: string;
  className?: string;
  children: ReactNode;
}

export function HUDPanel({ title, subtitle, className, children }: HUDPanelProps) {
  return (
    <section className={`glass-panel hud-outline hud-corners hud-glow ${className ?? ""}`}>
      <span className="corner-tr" aria-hidden="true" />
      <span className="corner-bl" aria-hidden="true" />
      <div className="border-b border-cyber-cyan/25 px-4 py-3">
        <div className="text-xs text-cyber-cyan/80 hud-title hud-text-glow">{title}</div>
        {subtitle ? <div className="mt-1 text-[11px] text-white/60">{subtitle}</div> : null}
      </div>
      <div className="px-4 py-4">{children}</div>
    </section>
  );
}
