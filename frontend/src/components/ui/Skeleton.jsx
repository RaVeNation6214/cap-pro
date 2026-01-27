import { clsx } from 'clsx'

export default function Skeleton({ className, ...props }) {
  return (
    <div
      className={clsx(
        'animate-pulse bg-dark-700/50 rounded',
        className
      )}
      {...props}
    />
  )
}

export function SkeletonText({ lines = 3, className }) {
  return (
    <div className={clsx('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={clsx(
            'h-4',
            i === lines - 1 ? 'w-3/4' : 'w-full'
          )}
        />
      ))}
    </div>
  )
}

export function SkeletonCard({ className }) {
  return (
    <div className={clsx('glass-card', className)}>
      <div className="flex items-center gap-4 mb-4">
        <Skeleton className="w-12 h-12 rounded-full" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-4 w-1/3" />
          <Skeleton className="h-3 w-1/2" />
        </div>
      </div>
      <SkeletonText lines={3} />
      <div className="mt-4 flex gap-2">
        <Skeleton className="h-8 w-20 rounded-lg" />
        <Skeleton className="h-8 w-20 rounded-lg" />
      </div>
    </div>
  )
}

export function SkeletonCode({ lines = 10, className }) {
  return (
    <div className={clsx('code-editor', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 py-1">
          <Skeleton className="w-8 h-4" />
          <Skeleton
            className="h-4"
            style={{ width: `${Math.random() * 40 + 30}%` }}
          />
        </div>
      ))}
    </div>
  )
}
