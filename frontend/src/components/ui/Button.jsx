import { motion } from 'framer-motion'
import { clsx } from 'clsx'

const variants = {
  primary: 'px-6 py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-700 hover:to-accent-700 focus:ring-2 focus:ring-primary-600/50 focus:outline-none transform hover:scale-105 active:scale-95 transition-all duration-200 ease-out shadow-lg shadow-primary-600/30 hover:shadow-primary-600/40',
  secondary: 'px-6 py-3 rounded-xl font-semibold bg-white/60 text-dark-800 border border-primary-300 hover:bg-white/80 hover:border-primary-400 focus:ring-2 focus:ring-primary-400/50 focus:outline-none transform hover:scale-105 active:scale-95 transition-all duration-200 ease-out',
  ghost: 'px-4 py-2 rounded-lg text-dark-700 hover:text-dark-900 hover:bg-white/60 transition-colors',
  danger: 'px-6 py-3 rounded-xl font-semibold bg-red-500/20 text-red-600 border border-red-500/30 hover:bg-red-500/30 transition-all',
}

const sizes = {
  sm: 'text-sm px-3 py-1.5',
  md: '',
  lg: 'text-lg px-8 py-4',
}

export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  className,
  disabled,
  loading,
  icon: Icon,
  ...props
}) {
  return (
    <motion.button
      whileHover={disabled ? {} : { scale: 1.02 }}
      whileTap={disabled ? {} : { scale: 0.98 }}
      disabled={disabled || loading}
      className={clsx(
        variants[variant],
        sizes[size],
        'flex items-center justify-center gap-2',
        'disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none',
        className
      )}
      {...props}
    >
      {loading ? (
        <svg
          className="animate-spin h-5 w-5"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
      ) : Icon ? (
        <Icon className="w-5 h-5" />
      ) : null}
      {children}
    </motion.button>
  )
}
