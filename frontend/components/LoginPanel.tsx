"use client";

import { useEffect, useState } from "react";
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
  type User,
} from "firebase/auth";
import { HUDPanel } from "./HUDPanel";
import { auth, firebaseEnabled } from "../lib/firebase";

export function LoginPanel() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!auth) {
      return;
    }
    return onAuthStateChanged(auth, (nextUser) => {
      setUser(nextUser);
    });
  }, []);

  const handleLogin = async () => {
    if (!auth) {
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setError(getAuthErrorMessage(err, "login"));
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async () => {
    if (!auth) {
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await createUserWithEmailAndPassword(auth, email, password);
    } catch (err) {
      setError(getAuthErrorMessage(err, "signup"));
    } finally {
      setLoading(false);
    }
  };

  const handleSignOut = async () => {
    if (!auth) {
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await signOut(auth);
    } catch (err) {
      setError(getAuthErrorMessage(err, "signout"));
    } finally {
      setLoading(false);
    }
  };

  if (!firebaseEnabled) {
    return (
      <HUDPanel title="User Login" subtitle="Firebase not configured">
        <div className="text-xs text-white/60">
          Add Firebase config env vars to enable authentication.
        </div>
      </HUDPanel>
    );
  }

  return (
    <HUDPanel title="User Login" subtitle="Authorized personnel only">
      <div className="space-y-4">
        {user ? (
          <>
            <div className="text-xs text-white/70">
              Signed in as <span className="text-white">{user.email}</span>
            </div>
            <button
              className="w-full rounded border border-cyber-cyan/60 bg-cyber-cyan/20 px-4 py-2 text-xs uppercase tracking-[0.2em] text-cyber-cyan hover:bg-cyber-cyan/30 disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleSignOut}
              disabled={loading}
            >
              Sign Out
            </button>
          </>
        ) : (
          <>
            <label className="block text-xs text-cyber-cyan/70">
              Email
              <div className="mt-2 flex items-center gap-2 rounded border border-cyber-cyan/40 bg-[#03070f] px-3 py-2 shadow-[0_0_18px_rgba(0,217,255,0.12)]">
                <span className="text-cyber-cyan/60">●</span>
                <input
                  type="email"
                  className="w-full bg-transparent text-xs text-white placeholder:text-white/40 focus:outline-none"
                  placeholder="name@company.com"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                />
              </div>
            </label>
            <label className="block text-xs text-cyber-cyan/70">
              Password
              <div className="mt-2 flex items-center gap-2 rounded border border-cyber-cyan/40 bg-[#03070f] px-3 py-2 shadow-[0_0_18px_rgba(0,217,255,0.12)]">
                <span className="text-cyber-cyan/60">●</span>
                <input
                  type="password"
                  className="w-full bg-transparent text-xs text-white placeholder:text-white/40 focus:outline-none"
                  placeholder="Enter password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              </div>
            </label>
            {error ? <div className="text-[10px] text-red-300">{error}</div> : null}
            <div className="flex gap-2">
              <button
                className="flex-1 rounded border border-cyber-cyan/60 bg-cyber-cyan/20 px-4 py-2 text-xs uppercase tracking-[0.2em] text-cyber-cyan hover:bg-cyber-cyan/30 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={handleLogin}
                disabled={loading || !email || !password}
              >
                Login
              </button>
              <button
                className="flex-1 rounded border border-cyber-cyan/40 px-4 py-2 text-xs uppercase tracking-[0.2em] text-cyber-cyan hover:bg-cyber-cyan/10 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={handleSignup}
                disabled={loading || !email || !password}
              >
                Sign Up
              </button>
            </div>
          </>
        )}
      </div>
    </HUDPanel>
  );
}

function getAuthErrorMessage(err: unknown, context: "login" | "signup" | "signout") {
  const fallback =
    context === "signup"
      ? "Signup failed. Please try again."
      : context === "signout"
        ? "Sign out failed. Please try again."
        : "Login failed. Please try again.";

  if (!err || typeof err !== "object") {
    return fallback;
  }

  const code = "code" in err ? String((err as { code?: string }).code) : "";

  switch (code) {
    case "auth/invalid-credential":
    case "auth/user-not-found":
    case "auth/wrong-password":
      return "Incorrect email or password.";
    case "auth/invalid-email":
      return "Please enter a valid email address.";
    case "auth/too-many-requests":
      return "Too many attempts. Please wait a moment and try again.";
    case "auth/email-already-in-use":
      return "That email is already registered. Try logging in instead.";
    case "auth/weak-password":
      return "Password is too weak. Use at least 6 characters.";
    default:
      return fallback;
  }
}
