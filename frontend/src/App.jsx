import { useState, useRef, useEffect } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Navbar from './components/layout/Navbar'
import Footer from './components/layout/Footer'
import Home from './pages/Home'
import Analyze from './pages/Analyze'
import Results from './pages/Results'
import { Bot, X, Send, Sparkles, ChevronDown } from 'lucide-react'
import { sendChat } from './services/api'

// ── Global Gemini chat FAB ─────────────────────────────────────────────────
function GlobalChat() {
  const [open, setOpen] = useState(false)
  const [msgs, setMsgs] = useState([
    { role: 'assistant', content: '👋 Hi! I\'m your Solidity security assistant. Ask me anything about smart contract vulnerabilities, Solidity best practices, or security auditing.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  async function send() {
    if (!input.trim() || loading) return
    const userMsg = { role: 'user', content: input.trim() }
    const history = [...msgs, userMsg]
    setMsgs(history)
    setInput('')
    setLoading(true)
    try {
      const res = await sendChat(history.filter(m => m.role !== 'system'))
      setMsgs(prev => [...prev, { role: 'assistant', content: res.reply }])
    } catch {
      setMsgs(prev => [...prev, { role: 'assistant', content: '❌ AI unavailable. Set GEMINI_API_KEY in backend/.env' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* FAB */}
      <button onClick={() => setOpen(o => !o)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg flex items-center justify-center transition-all hover:scale-105 active:scale-95">
        {open ? <ChevronDown size={22} /> : <Bot size={22} />}
      </button>

      {/* Chat panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="fixed bottom-24 right-6 z-50 w-[360px] rounded-2xl bg-white border border-gray-200 shadow-2xl overflow-hidden flex flex-col"
            style={{ maxHeight: 520 }}>

            {/* Header */}
            <div className="px-4 py-3 bg-indigo-600 flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                <Bot size={16} className="text-white" />
              </div>
              <div className="flex-1">
                <div className="text-white font-semibold text-sm flex items-center gap-1.5">
                  Security Assistant <Sparkles size={11} className="text-yellow-300" />
                </div>
                <div className="text-indigo-200 text-xs">Powered by Gemini 2.0 Flash</div>
              </div>
              <button onClick={() => setOpen(false)} className="text-white/70 hover:text-white">
                <X size={16} />
              </button>
            </div>

            {/* Quick prompts */}
            <div className="px-3 py-2 border-b border-gray-100 flex gap-1.5 overflow-x-auto scrollbar-none bg-gray-50">
              {['Fix reentrancy', 'Explain SWC-107', 'Best practices', 'Audit checklist'].map(q => (
                <button key={q} onClick={() => setInput(q)}
                  className="shrink-0 px-2.5 py-1 rounded-full bg-white border border-gray-200 text-gray-600 text-xs hover:border-indigo-300 hover:text-indigo-600 transition-colors whitespace-nowrap">
                  {q}
                </button>
              ))}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-3 space-y-3" style={{ minHeight: 0 }}>
              {msgs.map((m, i) => (
                <div key={i} className={`flex gap-2 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {m.role === 'assistant' && (
                    <div className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center shrink-0 mt-0.5">
                      <Bot size={12} className="text-indigo-600" />
                    </div>
                  )}
                  <div className={`max-w-[85%] rounded-xl px-3 py-2 text-xs leading-relaxed ${
                    m.role === 'user'
                      ? 'bg-indigo-600 text-white rounded-br-none'
                      : 'bg-gray-100 text-gray-800 rounded-bl-none'
                  }`}>
                    {m.content}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex gap-2">
                  <div className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center shrink-0">
                    <Bot size={12} className="text-indigo-600" />
                  </div>
                  <div className="bg-gray-100 rounded-xl px-3 py-2 flex gap-1 items-center">
                    {[0,1,2].map(i => (
                      <span key={i} className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: `${i*0.15}s` }} />
                    ))}
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>

            {/* Input */}
            <div className="p-3 border-t border-gray-100">
              <div className="flex gap-2">
                <input value={input} onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
                  placeholder="Ask about smart contract security…"
                  className="flex-1 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-xs text-gray-800 placeholder-gray-400 focus:outline-none focus:border-indigo-400 focus:bg-white transition-colors" />
                <button onClick={send} disabled={loading || !input.trim()}
                  className="w-8 h-8 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 flex items-center justify-center transition-colors shrink-0">
                  <Send size={13} className="text-white" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

// ── App ───────────────────────────────────────────────────────────────────
function App() {
  const location = useLocation()
  const isResults = location.pathname === '/results'

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Navbar />
      <main className="flex-1 pt-14">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Home />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </AnimatePresence>
      </main>
      {!isResults && <Footer />}
      <GlobalChat />
    </div>
  )
}

export default App
