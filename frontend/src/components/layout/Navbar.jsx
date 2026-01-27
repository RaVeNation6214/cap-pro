import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, Menu, X, Zap, FileCode, Info } from 'lucide-react'

const navLinks = [
  { path: '/', label: 'Home', icon: Shield },
  { path: '/analyze', label: 'Analyze', icon: FileCode },
  { path: '/about', label: 'About', icon: Info },
]

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const location = useLocation()

  return (
    <nav className="fixed top-0 left-0 right-0 z-50">
      <div className="glass border-b border-dark-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <motion.div
                whileHover={{ rotate: 15, scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                className="relative"
              >
                <Shield className="w-8 h-8 text-primary-400" />
                <motion.div
                  className="absolute inset-0 bg-primary-400/30 blur-xl rounded-full"
                  animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </motion.div>
              <div className="flex flex-col">
                <span className="text-lg font-bold gradient-text">
                  SmartAudit
                </span>
                <span className="text-xs text-dark-400 hidden sm:block">
                  AI Vulnerability Detection
                </span>
              </div>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-1">
              {navLinks.map((link) => {
                const Icon = link.icon
                const isActive = location.pathname === link.path

                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    className="relative px-4 py-2 rounded-lg transition-colors"
                  >
                    <motion.div
                      className="flex items-center gap-2"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Icon className={`w-4 h-4 ${isActive ? 'text-primary-400' : 'text-dark-400'}`} />
                      <span className={isActive ? 'text-primary-400 font-medium' : 'text-dark-300 hover:text-dark-100'}>
                        {link.label}
                      </span>
                    </motion.div>
                    {isActive && (
                      <motion.div
                        layoutId="navbar-indicator"
                        className="absolute inset-0 bg-primary-500/10 rounded-lg border border-primary-500/20"
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
                  className="btn-primary flex items-center gap-2"
                >
                  <Zap className="w-4 h-4" />
                  Start Analysis
                </motion.button>
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <motion.button
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden p-2 rounded-lg bg-dark-800/50 border border-dark-700"
            >
              {isOpen ? (
                <X className="w-5 h-5 text-dark-300" />
              ) : (
                <Menu className="w-5 h-5 text-dark-300" />
              )}
            </motion.button>
          </div>
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
            className="md:hidden glass border-b border-dark-700/50"
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
                    className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-primary-500/10 text-primary-400 border border-primary-500/20'
                        : 'text-dark-300 hover:bg-dark-800/50'
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
                className="btn-primary w-full flex items-center justify-center gap-2 mt-4"
              >
                <Zap className="w-4 h-4" />
                Start Analysis
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  )
}
