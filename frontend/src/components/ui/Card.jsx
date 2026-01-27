import { motion } from 'framer-motion'
import { clsx } from 'clsx'

export default function Card({
  children,
  className,
  hover = true,
  gradient = false,
  ...props
}) {
  return (
    <motion.div
      whileHover={hover ? { y: -4, scale: 1.01 } : {}}
      transition={{ duration: 0.2 }}
      className={clsx(
        'glass-card',
        gradient && 'gradient-border',
        hover && 'cursor-pointer',
        className
      )}
      {...props}
    >
      {children}
    </motion.div>
  )
}

export function CardHeader({ children, className }) {
  return (
    <div className={clsx('mb-4', className)}>
      {children}
    </div>
  )
}

export function CardTitle({ children, className }) {
  return (
    <h3 className={clsx('text-lg font-semibold text-dark-100', className)}>
      {children}
    </h3>
  )
}

export function CardDescription({ children, className }) {
  return (
    <p className={clsx('text-sm text-dark-400 mt-1', className)}>
      {children}
    </p>
  )
}

export function CardContent({ children, className }) {
  return (
    <div className={clsx(className)}>
      {children}
    </div>
  )
}

export function CardFooter({ children, className }) {
  return (
    <div className={clsx('mt-4 pt-4 border-t border-dark-700/50', className)}>
      {children}
    </div>
  )
}
