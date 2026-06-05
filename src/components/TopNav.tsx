"use client";

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Shield, Sun, Moon } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';

export default function TopNav() {
  const { isDark, toggle: toggleTheme } = useTheme();
  const pathname = usePathname();

  return (
    <header className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ${isDark ? 'bg-[#050505]/80' : 'bg-[#f8f9fb]/80'} backdrop-blur-md border-b ${isDark ? 'border-zinc-900' : 'border-zinc-200'}`}>
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 font-bold text-lg group">
          <Shield className="w-5 h-5 text-purple-500 group-hover:scale-110 transition-transform" />
          <span className="bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            DF Forensics
          </span>
        </Link>
        
        {/* Links */}
        <nav className="flex items-center gap-6">
          <Link 
            href="/" 
            className={`text-sm font-medium transition-colors ${
              pathname === '/' 
                ? (isDark ? 'text-white' : 'text-zinc-900')
                : (isDark ? 'text-zinc-500 hover:text-white' : 'text-zinc-500 hover:text-zinc-900')
            }`}
          >
            Detetor
          </Link>
          <Link 
            href="/learn" 
            className={`text-sm font-medium transition-colors ${
              pathname === '/learn' 
                ? (isDark ? 'text-white' : 'text-zinc-900')
                : (isDark ? 'text-zinc-500 hover:text-white' : 'text-zinc-500 hover:text-zinc-900')
            }`}
          >
            Informação
          </Link>
          <Link
            href="/game"
            className={`text-sm font-medium transition-colors ${
              pathname === '/game'
                ? (isDark ? 'text-white' : 'text-zinc-900')
                : (isDark ? 'text-zinc-500 hover:text-white' : 'text-zinc-500 hover:text-zinc-900')
            }`}
          >
            Desafio
          </Link>
          <Link
            href="/historico"
            className={`text-sm font-medium transition-colors ${
              pathname === '/historico'
                ? (isDark ? 'text-white' : 'text-zinc-900')
                : (isDark ? 'text-zinc-500 hover:text-white' : 'text-zinc-500 hover:text-zinc-900')
            }`}
          >
            Histórico
          </Link>
          <Link
            href="/download"
            className={`text-sm font-medium transition-colors ${
              pathname === '/download'
                ? (isDark ? 'text-white' : 'text-zinc-900')
                : (isDark ? 'text-zinc-500 hover:text-white' : 'text-zinc-500 hover:text-zinc-900')
            }`}
          >
            Download
          </Link>
        </nav>

        {/* Actions */}
        <div className="flex items-center gap-4">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-full glass hover:scale-110 transition-all duration-300 group"
            title={isDark ? "Mudar para modo claro" : "Mudar para modo escuro"}
          >
            {isDark ? (
              <Sun className="w-4 h-4 text-yellow-400 group-hover:rotate-45 transition-transform duration-300" />
            ) : (
              <Moon className="w-4 h-4 text-purple-500 group-hover:-rotate-12 transition-transform duration-300" />
            )}
          </button>
        </div>

      </div>
    </header>
  );
}
