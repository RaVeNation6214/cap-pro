import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, FileCode, Home, BarChart3, Menu, X, Zap } from 'lucide-react'

const NAV_LINKS = [
  { path: '/',        label: 'Home',    icon: Home },
  { path: '/analyze', label: 'Analyze', icon: FileCode },
  { path: '/results', label: 'Results', icon: BarChart3 },
]

export default function Navbar() {
  const [open, setOpen] = useState(false)
  const location = useLocation()
  const hasResults = typeof window !== 'undefined' && !!sessionStorage.getItem('analysisResult')
  const visibleLinks = NAV_LINKS.filter(l => l.path !== '/results' || hasResults)

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-screen-xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">

          {/* Logo */}
          <Link to="/" className="flex items-center gap-2" onClick={() => setOpen(false)}>
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-sm">
              <Shield className="w-4 h-4 text-white" />
            </div>
            <span className="text-base font-bold text-gray-900 tracking-tight">SmartAudit</span>
            <span className="hidden sm:block text-xs text-gray-400 font-medium border border-gray-200 rounded px-1.5 py-0.5 bg-gray-50">AI</span>
          </Link>

          {/* Desktop links */}
          <div className="hidden md:flex items-center gap-1">
            {visibleLinks.map(({ path, label, icon: Icon }) => {
              const active = location.pathname === path
              return (
                <Link key={path} to={path}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    active
                      ? 'bg-indigo-50 text-indigo-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}>
                  <Icon size={14} />
                  {label}
                </Link>
              )
            })}
          </div>

          <div className="flex items-center gap-2">
            <Link to="/analyze"
              className="hidden md:flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold transition-colors shadow-sm">
              <Zap size={13} />
              Analyze Contract
            </Link>
            <button onClick={() => setOpen(o => !o)}
              className="md:hidden p-2 rounded-lg text-gray-500 hover:bg-gray-100">
              {open ? <X size={18} /> : <Menu size={18} />}
            </button>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.15 }}
            className="md:hidden border-t border-gray-100 bg-white overflow-hidden">
            <div className="px-4 py-3 space-y-1">
              {visibleLinks.map(({ path, label, icon: Icon }) => {
                const active = location.pathname === path
                return (
                  <Link key={path} to={path} onClick={() => setOpen(false)}
                    className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium ${
                      active ? 'bg-indigo-50 text-indigo-700' : 'text-gray-700 hover:bg-gray-50'
                    }`}>
                    <Icon size={15} />{label}
                  </Link>
                )
              })}
              <Link to="/analyze" onClick={() => setOpen(false)}
                className="flex items-center justify-center gap-2 mt-2 px-4 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-semibold">
                <Zap size={14} />Analyze Contract
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  )
}
