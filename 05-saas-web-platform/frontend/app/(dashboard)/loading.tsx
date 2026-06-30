export default function DashboardLoading() {
  return (
    <div className="p-8">
      <div className="animate-pulse space-y-6">
        {/* Page header skeleton */}
        <div className="space-y-2">
          <div className="h-8 bg-muted rounded w-48"></div>
          <div className="h-4 bg-muted rounded w-96"></div>
        </div>

        {/* Stats cards skeleton */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="border rounded-lg p-6">
              <div className="h-4 bg-muted rounded w-24 mb-2"></div>
              <div className="h-8 bg-muted rounded w-16"></div>
            </div>
          ))}
        </div>

        {/* Content skeleton */}
        <div className="border rounded-lg p-6">
          <div className="h-6 bg-muted rounded w-32 mb-4"></div>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-center gap-4">
                <div className="h-10 w-10 bg-muted rounded-full"></div>
                <div className="flex-1 space-y-2">
                  <div className="h-4 bg-muted rounded w-1/3"></div>
                  <div className="h-3 bg-muted rounded w-1/2"></div>
                </div>
                <div className="h-8 bg-muted rounded w-20"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
