import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, Menu, X, FileCode } from 'lucide-react'

const navLinks = [
  { path: '/', label: 'Home', icon: Shield },
  { path: '/analyze', label: 'Analyze', icon: FileCode },
]

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const location = useLocation()

  return (
    <nav className="fixed top-4 left-0 right-0 z-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14 rounded-full bg-white/80 backdrop-blur border border-white/70 shadow-[0_20px_40px_rgba(120,110,220,0.2)] px-4 sm:px-6">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <motion.div
                whileHover={{ rotate: 15, scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                className="relative"
              >
                <Shield className="w-7 h-7 text-primary-600" />
              </motion.div>
              <div className="flex flex-col">
                <span className="text-sm sm:text-base font-bold text-dark-900 tracking-wide">
                  SmartAudit
                </span>
                <span className="text-[10px] text-dark-500 hidden sm:block">AI Vulnerability Detection</span>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-6">
              {navLinks.map((link) => {
                const Icon = link.icon
                const isActive = location.pathname === link.path

                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    className="relative px-2 py-1 text-sm text-dark-600 hover:text-dark-900 transition-colors"
                  >
                    <motion.div
                      className="flex items-center gap-2"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Icon className={`w-4 h-4 ${isActive ? 'text-primary-600' : 'text-dark-500'}`} />
                      <span className={isActive ? 'text-primary-700 font-semibold' : 'text-dark-600'}>
                        {link.label}
                      </span>
                    </motion.div>
                    {isActive && (
                      <motion.div
                        layoutId="navbar-indicator"
                        className="absolute -bottom-2 left-1/2 h-1 w-6 -translate-x-1/2 rounded-full bg-primary-500"
                        initial={false}
                        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                      />
                    )}
                  </Link>
                )
              })}
            </div>

            {/* CTA Button */}
            <div className="hidden md:block">
              <Link to="/analyze">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="rounded-md bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow-[0_12px_24px_rgba(79,70,229,0.3)] transition hover:bg-primary-700"
                >
                  Open Analyzer
                </motion.button>
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden p-2 rounded-lg bg-white/80 border border-primary-200"
            >
              {isOpen ? (
                <X className="w-5 h-5 text-dark-700" />
              ) : (
                <Menu className="w-5 h-5 text-dark-700" />
              )}
            </motion.button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden mt-3 mx-4 rounded-2xl bg-white/90 border border-white/70 shadow-[0_20px_40px_rgba(120,110,220,0.18)]"
          >
            <div className="px-4 py-4 space-y-2">
              {navLinks.map((link) => {
                const Icon = link.icon
                const isActive = location.pathname === link.path

                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    onClick={() => setIsOpen(false)}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-colors ${isActive
                      ? 'bg-primary-500/10 text-primary-700 border border-primary-500/20'
                      : 'text-dark-700 hover:bg-primary-50'
                      }`}
                  >
                    <Icon className="w-5 h-5" />
                    {link.label}
                  </Link>
                )
              })}
              <Link
                to="/analyze"
                onClick={() => setIsOpen(false)}
                className="w-full flex items-center justify-center gap-2 mt-4 rounded-md bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow-[0_12px_24px_rgba(79,70,229,0.3)]"
              >
                Open Analyzer
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  )
}
