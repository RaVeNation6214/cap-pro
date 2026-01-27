import { clsx } from 'clsx'

const variants = {
  default: 'bg-dark-700 text-dark-200',
  primary: 'bg-primary-500/20 text-primary-400 border border-primary-500/30',
  secondary: 'bg-accent-500/20 text-accent-400 border border-accent-500/30',
  success: 'bg-green-500/20 text-green-400 border border-green-500/30',
  warning: 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
  danger: 'bg-red-500/20 text-red-400 border border-red-500/30',
  safe: 'risk-safe',
  low: 'risk-low',
  medium: 'risk-medium',
  high: 'risk-high',
  critical: 'risk-critical',
}

const sizes = {
  sm: 'text-xs px-2 py-0.5',
  md: 'text-sm px-3 py-1',
  lg: 'text-base px-4 py-1.5',
}

export default function Badge({
  children,
  variant = 'default',
  size = 'md',
  className,
  ...props
}) {
  return (
    <span
      className={clsx(
        'inline-flex items-center font-medium rounded-full',
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}
