import { motion } from 'framer-motion'
import { Shield, Github, Twitter, Heart } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="relative mt-auto">
      {/* Gradient divider */}
      <div className="h-px bg-gradient-to-r from-transparent via-primary-500/50 to-transparent" />

      <div className="glass border-t border-dark-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {/* Brand */}
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center gap-3 mb-4">
                <Shield className="w-8 h-8 text-primary-400" />
                <span className="text-xl font-bold gradient-text">SmartAudit</span>
              </div>
              <p className="text-dark-400 max-w-md">
                AI-powered smart contract vulnerability detection using Hierarchical
                Transformer architecture. Secure your blockchain applications with
                cutting-edge machine learning.
              </p>
            </div>

            {/* Links */}
            <div>
              <h3 className="text-sm font-semibold text-dark-200 uppercase tracking-wider mb-4">
                Resources
              </h3>
              <ul className="space-y-2">
                {['Documentation', 'API Reference', 'Research Paper', 'GitHub'].map((item) => (
                  <li key={item}>
                    <a href="#" className="text-dark-400 hover:text-primary-400 transition-colors">
                      {item}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Vulnerability Types */}
            <div>
              <h3 className="text-sm font-semibold text-dark-200 uppercase tracking-wider mb-4">
                Detection
              </h3>
              <ul className="space-y-2">
                {['Reentrancy', 'Access Control', 'Arithmetic', 'Unchecked Calls'].map((item) => (
                  <li key={item}>
                    <span className="text-dark-400">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Bottom bar */}
          <div className="mt-8 pt-8 border-t border-dark-700/50 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-dark-500 text-sm flex items-center gap-1">
              Made with <Heart className="w-4 h-4 text-red-500 animate-pulse" /> for B.Tech Capstone 2026
            </p>

            <div className="flex items-center gap-4">
              {[Github, Twitter].map((Icon, index) => (
                <motion.a
                  key={index}
                  href="#"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="p-2 rounded-lg bg-dark-800/50 border border-dark-700 text-dark-400 hover:text-primary-400 transition-colors"
                >
                  <Icon className="w-5 h-5" />
                </motion.a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
