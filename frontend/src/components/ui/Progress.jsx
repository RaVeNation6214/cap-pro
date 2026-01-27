import { motion } from 'framer-motion'
import { clsx } from 'clsx'

const colorVariants = {
  default: 'bg-primary-500',
  safe: 'bg-risk-safe',
  low: 'bg-risk-low',
  medium: 'bg-risk-medium',
  high: 'bg-risk-high',
  critical: 'bg-risk-critical',
}

export default function Progress({
  value = 0,
  max = 100,
  variant = 'default',
  showLabel = false,
  label,
  size = 'md',
  className,
  animate = true,
}) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))

  const heights = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  }

  return (
    <div className={clsx('w-full', className)}>
      {(showLabel || label) && (
        <div className="flex justify-between mb-1.5">
          <span className="text-sm text-dark-300">{label}</span>
          {showLabel && (
            <span className="text-sm font-medium text-dark-200">
              {percentage.toFixed(0)}%
            </span>
          )}
        </div>
      )}
      <div className={clsx('w-full bg-dark-800 rounded-full overflow-hidden', heights[size])}>
        <motion.div
          initial={animate ? { width: 0 } : false}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={clsx('h-full rounded-full', colorVariants[variant])}
        />
      </div>
    </div>
  )
}

export function MultiProgress({ segments, className }) {
  const total = segments.reduce((sum, s) => sum + s.value, 0)

  return (
    <div className={clsx('w-full', className)}>
      <div className="h-3 bg-dark-800 rounded-full overflow-hidden flex">
        {segments.map((segment, index) => (
          <motion.div
            key={index}
            initial={{ width: 0 }}
            animate={{ width: `${(segment.value / total) * 100}%` }}
            transition={{ duration: 0.8, delay: index * 0.1, ease: 'easeOut' }}
            className={clsx('h-full', colorVariants[segment.variant || 'default'])}
            title={segment.label}
          />
        ))}
      </div>
      {segments.some(s => s.label) && (
        <div className="flex flex-wrap gap-4 mt-2">
          {segments.map((segment, index) => (
            <div key={index} className="flex items-center gap-2">
              <div className={clsx('w-3 h-3 rounded-full', colorVariants[segment.variant || 'default'])} />
              <span className="text-sm text-dark-400">{segment.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
