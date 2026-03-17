import { clsx } from 'clsx'

const variants = {
  default: 'bg-white/80 text-slate-700 border border-slate-300',
  primary: 'bg-primary-100 text-primary-700 border border-primary-300',
  secondary: 'bg-accent-100 text-accent-700 border border-accent-300',
  success: 'bg-green-100 text-green-800 border border-green-300',
  warning: 'bg-amber-100 text-amber-800 border border-amber-300',
  danger: 'bg-red-100 text-red-800 border border-red-300',
  safe: 'bg-green-100 text-green-800 border border-green-300',
  low: 'bg-lime-100 text-lime-800 border border-lime-300',
  medium: 'bg-amber-100 text-amber-800 border border-amber-300',
  high: 'bg-orange-100 text-orange-800 border border-orange-300',
  critical: 'bg-red-100 text-red-800 border border-red-300',
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
