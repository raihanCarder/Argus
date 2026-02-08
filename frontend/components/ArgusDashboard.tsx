"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { HUDPanel } from "./HUDPanel";
import { LoginPanel } from "./LoginPanel";
import { AIChat } from "./AIChat";
import { matchStartup, type MatchResult, type Signal } from "../lib/api";

const Earth3D = dynamic(() => import("./Earth3D").then((mod) => mod.Earth3D), {
  ssr: false,
});

export function ArgusDashboard() {
  const [startupDescription, setStartupDescription] = useState("");
  const [matches, setMatches] = useState<MatchResult[]>([]);
  const [activeMatch, setActiveMatch] = useState<MatchResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);

  const signals = useMemo(
    () => matches.map((match) => match.signal),
    [matches],
  );

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const storedMatches = window.localStorage.getItem("argus.matches");
    const storedStartup = window.localStorage.getItem("argus.startup");
    const storedHistory = window.localStorage.getItem("argus.history");
    if (storedMatches) {
      try {
        const parsed = JSON.parse(storedMatches) as MatchResult[];
        if (Array.isArray(parsed)) {
          setMatches(parsed);
          setActiveMatch(parsed[0] ?? null);
          setHasSearched(parsed.length > 0);
        }
      } catch {
        // ignore malformed cache
      }
    }
    if (storedStartup) {
      setStartupDescription(storedStartup);
    }
    if (storedHistory) {
      try {
        const parsed = JSON.parse(storedHistory) as string[];
        if (Array.isArray(parsed)) {
          setSearchHistory(parsed);
        }
      } catch {
        // ignore malformed cache
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem("argus.matches", JSON.stringify(matches));
  }, [matches]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem("argus.startup", startupDescription);
  }, [startupDescription]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem("argus.history", JSON.stringify(searchHistory));
  }, [searchHistory]);

  const handleSearch = async () => {
    if (!startupDescription.trim()) {
      return;
    }
    setIsLoading(true);
    setError(null);
    setHasSearched(true);
    try {
      const response = await matchStartup(startupDescription.trim());
      setMatches((prev) => {
        const merged = new Map<number, MatchResult>();
        prev.forEach((item) => merged.set(item.signal.id, item));
        response.matches.forEach((item) => merged.set(item.signal.id, item));
        return Array.from(merged.values());
      });
      setActiveMatch(response.matches[0] ?? null);
      setSearchHistory((prev) => {
        const next = [startupDescription.trim(), ...prev];
        return Array.from(new Set(next)).slice(0, 8);
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Match request failed";
      setError(message);
      setMatches([]);
      setActiveMatch(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignalClick = (signal: Signal) => {
    const match = matches.find((item) => item.signal.id === signal.id);
    setActiveMatch(match ?? null);
  };

  return (
    <main className="relative min-h-screen text-white overflow-hidden">
      <div className="absolute inset-0 hud-grid opacity-50" />
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute left-8 top-8 h-2 w-20 border-t border-cyber-cyan/50" />
        <div className="absolute right-10 top-10 h-2 w-24 border-t border-cyber-cyan/50" />
        <div className="absolute left-8 bottom-10 h-2 w-24 border-b border-cyber-cyan/50" />
        <div className="absolute right-10 bottom-8 h-2 w-20 border-b border-cyber-cyan/50" />
        <div className="glitch-line" />
      </div>
      <div className="absolute inset-x-0 top-0 h-full scanline opacity-35" />

      <header className="relative z-10 flex items-center justify-between px-10 py-6">
        <div>
          <div className="hud-title text-xs text-cyber-cyan hud-text-glow">ARGUS</div>
          <div className="text-[10px] text-white/50">
            Matches startups to relevant government signals in real-time.
          </div>
        </div>
        <div className="flex items-center gap-6 text-[10px] hud-mono text-white/60">
          <div className="flex items-center gap-2">
            <span className="hud-dot" />
            {isLoading ? "Searching" : "Tracking"}
          </div>
          <div>Signals: {signals.length}</div>
          <div>Status: Stable</div>
        </div>
      </header>

      <div className="relative z-10 grid h-[calc(100vh-96px)] grid-cols-[280px_minmax(0,1fr)_320px] gap-6 px-10 pb-10">
        <aside className="flex h-full flex-col gap-6">
          <LoginPanel />
          <AIChat
            value={startupDescription}
            onChange={setStartupDescription}
            onSubmit={handleSearch}
            loading={isLoading}
            error={error}
          />
        </aside>

        <section className="relative flex h-full flex-col items-center justify-center">
          <div className="h-full w-full">
            <Earth3D signals={signals} onSignalClick={handleSignalClick} />
          </div>
          <div className="absolute bottom-6 flex w-full items-center justify-between px-6 text-[10px] text-white/50 hud-mono">
            <div>LAT: 40.7128° N</div>
            <div>LONG: 74.0060° W</div>
            <div className="text-cyber-cyan/70">STABLE</div>
          </div>
        </section>

        <aside className="flex h-full flex-col gap-6">
          {activeMatch ? (
            <HUDPanel
              title="Signal Detail"
              subtitle={`${activeMatch.signal.city || activeMatch.signal.country || "Unknown"}${activeMatch.signal.state ? `, ${activeMatch.signal.state}` : ""} · ${activeMatch.signal.category}`}
            >
              <div className="space-y-4 text-sm">
                <div className="text-lg font-semibold text-white">
                  {activeMatch.signal.title}
                </div>
                <p className="text-white/70">
                  {activeMatch.signal.description}
                </p>
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <div className="text-white/40">Budget</div>
                    <div className="hud-mono text-white">
                      {activeMatch.signal.budget
                        ? `$${(activeMatch.signal.budget / 1000000).toFixed(1)}M`
                        : "N/A"}
                    </div>
                  </div>
                  <div>
                    <div className="text-white/40">Timeline</div>
                    <div className="text-white">
                      {activeMatch.signal.timeline}
                    </div>
                  </div>
                </div>
                <div className="text-xs text-white/60">
                  Match Score: {Math.round(activeMatch.score * 100)}%
                </div>
                <div>
                  <div className="text-white/40 text-xs">Stakeholders</div>
                  <ul className="mt-2 space-y-1 text-xs text-white/80">
                    {activeMatch.signal.stakeholders.map((person) => (
                      <li key={person}>• {person}</li>
                    ))}
                  </ul>
                </div>
                <div className="flex justify-end">
                  <button
                    className="rounded border border-cyber-cyan/60 bg-cyber-cyan/20 px-3 py-1 text-[10px] uppercase tracking-[0.2em] text-cyber-cyan hover:bg-cyber-cyan/30"
                    onClick={() => setActiveMatch(null)}
                    aria-label="Close signal detail"
                  >
                    Back
                  </button>
                </div>
              </div>
            </HUDPanel>
          ) : (
            <HUDPanel title="Signal Detail" subtitle="Awaiting selection">
              <div className="text-sm text-white/60">
                {isLoading
                  ? "Matching signals..."
                  : signals.length > 0
                    ? "Select a signal pin to view government entities."
                    : hasSearched
                      ? "No matches found. Try another description."
                      : "Select a signal pin to view government entities."}
              </div>
            </HUDPanel>
          )}
        </aside>
      </div>
    </main>
  );
}
