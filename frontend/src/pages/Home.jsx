import { Link } from 'react-router-dom'
import { CheckCircle, Shield, Code, BarChart3, GitBranch } from 'lucide-react'

export default function Home() {
  return (
    <div className="pt-24 bg-gradient-to-b from-[#efeefe] via-[#f7f6ff] to-white">
      <section className="px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto rounded-[32px] bg-gradient-to-br from-[#bdb8ff] via-[#d6d3ff] to-[#9a95ff] p-6 sm:p-8 lg:p-10 shadow-[0_35px_80px_rgba(88,73,218,0.35)]">
          <div className="grid gap-10 lg:grid-cols-[1.05fr_0.95fr] items-center">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-dark-600 mb-4">SmartAudit</p>
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-[1.05] text-dark-900">
                SmartAudit AI:
                <span className="block text-dark-900">Security that</span>
                <span className="block text-primary-700">starts in</span>
                <span className="block text-primary-700">development</span>
              </h1>
              <p className="mt-6 text-base sm:text-lg text-dark-700 max-w-xl">
                SmartAudit is our capstone interface for the backend detection service. It runs
                contract analysis, surfaces model confidence, and highlights vulnerable lines for
                fast iteration during development.
              </p>
              <div className="mt-8">
                <Link
                  to="/analyze"
                  className="inline-flex items-center gap-2 rounded-lg bg-primary-600 px-5 py-3 text-sm sm:text-base font-semibold text-white shadow-[0_10px_20px_rgba(79,70,229,0.35)] transition hover:-translate-y-0.5 hover:bg-primary-700"
                >
                  Open Analyzer
                  <span className="text-white/80">→</span>
                </Link>
              </div>
              <div className="mt-10 flex items-center gap-3 text-sm text-dark-700">
                <CheckCircle className="w-4 h-4 text-dark-800" />
                Capstone demo connected to the backend inference pipeline
              </div>
            </div>

            <div className="relative">
              <div className="rounded-2xl bg-[#1b1d33] border border-white/10 shadow-[0_30px_60px_rgba(20,16,50,0.45)] overflow-hidden">
                <div className="flex items-center justify-between px-5 py-4 border-b border-white/10 text-xs text-white/60">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-400/80" />
                    <span className="w-2.5 h-2.5 rounded-full bg-yellow-400/80" />
                    <span className="w-2.5 h-2.5 rounded-full bg-green-400/80" />
                  </div>
                  <span>SmartAudit Console</span>
                  <span className="opacity-0">controls</span>
                </div>
                <div className="px-6 py-5 text-white/80 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-white font-semibold">Audit Overview</span>
                    <span className="text-xs text-white/50">Last 30 days</span>
                  </div>
                  <div className="mt-4 space-y-3">
                    {[
                      { label: 'Reentrancy', value: 'Critical', color: 'bg-red-400' },
                      { label: 'Access Control', value: 'High', color: 'bg-orange-400' },
                      { label: 'Unchecked Calls', value: 'Medium', color: 'bg-yellow-400' },
                    ].map((row) => (
                      <div key={row.label} className="flex items-center justify-between rounded-lg bg-white/5 px-4 py-3">
                        <span className="text-sm text-white/70">{row.label}</span>
                        <span className="flex items-center gap-2 text-xs text-white/60">
                          <span className={`h-2 w-2 rounded-full ${row.color}`} />
                          {row.value}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="mt-5 flex items-center gap-3 rounded-lg bg-white/5 px-4 py-3 text-xs text-white/70">
                    <Shield className="w-4 h-4 text-primary-300" />
                    Model confidence: 89% accuracy across 4 classes
                  </div>
                </div>
              </div>

              <div className="absolute -left-6 -bottom-8 w-[260px] rounded-xl bg-white/95 px-4 py-3 shadow-[0_18px_45px_rgba(60,45,160,0.25)]">
                <div className="text-[10px] uppercase tracking-[0.2em] text-dark-500">Latest run</div>
                <div className="mt-2 text-xs text-dark-700 flex items-center gap-2">
                  <Code className="w-4 h-4 text-primary-600" />
                  vault.sol analyzed in 7.2s
                </div>
                <div className="mt-2 flex items-center gap-2 text-[11px] text-dark-500">
                  <BarChart3 className="w-4 h-4 text-primary-500" />
                  3 findings, 1 critical
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-14">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="rounded-[24px] bg-white/70 border border-white/60 shadow-[0_20px_40px_rgba(145,137,220,0.2)] px-6 py-6">
            <div className="flex flex-wrap items-center justify-center gap-6 text-xs sm:text-sm text-dark-500 font-semibold tracking-wide">
              <span className="text-dark-400">Dataset: SmartBugs</span>
              <span className="text-dark-400">SWC Registry</span>
              <span className="text-dark-400">Solidity Samples</span>
              <span className="text-dark-400">DeFi Benchmarks</span>
              <span className="text-dark-400">Audit Patterns</span>
              <span className="text-dark-400">Custom Test Set</span>
            </div>
          </div>
        </div>
      </section>

      <section className="py-6">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid gap-12 lg:grid-cols-[0.9fr_1.1fr] items-center">
            <div className="relative rounded-2xl bg-[#1c1f38] p-6 shadow-[0_25px_60px_rgba(30,27,75,0.45)]">
              <div className="flex items-center justify-between text-white/70 text-xs mb-4">
                <span className="uppercase tracking-[0.2em]">Run history</span>
                <span>smart-audit-03</span>
              </div>
              <div className="space-y-3 text-sm text-white/80">
                {[
                  '0x9d2b...f91e • 12,042 lines',
                  '0xb8a4...aa11 • 7,210 lines',
                  '0xe91f...73b2 • 3,952 lines',
                ].map((item) => (
                  <div key={item} className="flex items-center justify-between rounded-lg bg-white/5 px-4 py-3">
                    <span>{item}</span>
                    <span className="text-xs text-primary-200">In review</span>
                  </div>
                ))}
              </div>
              <div className="mt-5 rounded-lg bg-white/10 px-4 py-3 text-xs text-white/70">
                Security review started • 4 critical issues prioritized
              </div>
            </div>

            <div>
              <p className="text-sm font-semibold text-primary-600 mb-3">Capstone objectives</p>
              <h2 className="text-3xl sm:text-4xl font-bold text-dark-900 mb-6">
                Bring research-grade detection to every analysis
              </h2>
              <div className="space-y-4 text-dark-700">
                {[
                  'Developer-friendly insights with attention heatmaps and clear root causes.',
                  'Backend-powered scoring with explainable confidence values.',
                  'Actionable output across 4 critical vulnerability classes.',
                ].map((item) => (
                  <div key={item} className="flex items-start gap-3">
                    <CheckCircle className="w-5 h-5 text-primary-600 mt-0.5" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
              <div className="mt-6 flex items-center gap-3 text-sm text-dark-500">
                <GitBranch className="w-4 h-4 text-primary-500" />
                Built to demonstrate the backend model in a clean UI
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="rounded-2xl bg-white border border-[#ebe9ff] shadow-[0_25px_60px_rgba(122,109,216,0.18)] p-8">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-primary-100 flex items-center justify-center text-primary-600 font-semibold">
                  Cap
                </div>
                <div>
                  <p className="text-sm font-semibold text-dark-900">Capstone Project</p>
                  <p className="text-xs text-dark-500">B.Tech 2026 • Smart Contract Security</p>
                </div>
              </div>
              <div className="text-xs text-dark-500">“A compact UI to validate backend model performance end‑to‑end.”</div>
            </div>
            <p className="mt-6 text-sm text-dark-600 leading-relaxed max-w-3xl">
              This interface focuses on clarity: input a contract, run the backend model, and
              review detected vulnerabilities with confidence scores. It is built to demonstrate
              the full pipeline for our capstone evaluation.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
