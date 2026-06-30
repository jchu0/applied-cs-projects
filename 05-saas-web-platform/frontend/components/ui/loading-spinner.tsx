import { cn } from '@/lib/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeClasses = {
  sm: 'w-4 h-4 border-2',
  md: 'w-8 h-8 border-3',
  lg: 'w-12 h-12 border-4',
};

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  return (
    <div className={cn('relative', className)}>
      <div className={cn('border-muted rounded-full', sizeClasses[size])}></div>
      <div
        className={cn(
          'absolute top-0 left-0 border-primary border-t-transparent rounded-full animate-spin',
          sizeClasses[size]
        )}
      ></div>
    </div>
  );
}

interface LoadingOverlayProps {
  message?: string;
}

export function LoadingOverlay({ message = 'Loading...' }: LoadingOverlayProps) {
  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <LoadingSpinner size="lg" />
        <p className="text-sm text-muted-foreground">{message}</p>
      </div>
    </div>
  );
}

interface LoadingButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  loading?: boolean;
  children: React.ReactNode;
}

export function LoadingButton({ loading, children, disabled, className, ...props }: LoadingButtonProps) {
  return (
    <button
      disabled={loading || disabled}
      className={cn(
        'relative inline-flex items-center justify-center',
        loading && 'cursor-not-allowed',
        className
      )}
      {...props}
    >
      {loading && (
        <LoadingSpinner size="sm" className="absolute" />
      )}
      <span className={cn(loading && 'invisible')}>{children}</span>
    </button>
  );
}

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div className={cn('bg-muted rounded animate-pulse', className)} />
  );
}

interface CardSkeletonProps {
  lines?: number;
}

export function CardSkeleton({ lines = 3 }: CardSkeletonProps) {
  return (
    <div className="border rounded-lg p-6 space-y-4">
      <Skeleton className="h-6 w-1/3" />
      {[...Array(lines)].map((_, i) => (
        <Skeleton key={i} className="h-4 w-full" />
      ))}
    </div>
  );
}

interface TableSkeletonProps {
  rows?: number;
  columns?: number;
}

export function TableSkeleton({ rows = 5, columns = 4 }: TableSkeletonProps) {
  return (
    <div className="border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-muted/50 p-4 flex gap-4">
        {[...Array(columns)].map((_, i) => (
          <Skeleton key={i} className="h-4 flex-1" />
        ))}
      </div>
      {/* Rows */}
      {[...Array(rows)].map((_, i) => (
        <div key={i} className="p-4 border-t flex gap-4">
          {[...Array(columns)].map((_, j) => (
            <Skeleton key={j} className="h-4 flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}
