import { motion } from 'framer-motion'
import { clsx } from 'clsx'

const variants = {
  primary: 'btn-primary',
  secondary: 'btn-secondary',
  ghost: 'px-4 py-2 rounded-lg text-dark-300 hover:text-white hover:bg-dark-800/50 transition-colors',
  danger: 'px-6 py-3 rounded-xl font-semibold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-all',
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
