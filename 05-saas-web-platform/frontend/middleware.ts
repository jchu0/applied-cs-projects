import { NextRequest, NextResponse } from 'next/server'

// Routes that require authentication
const protectedRoutes = ['/dashboard', '/settings', '/billing', '/team']

// Routes that should redirect to dashboard if already authenticated
const authRoutes = ['/login', '/signup', '/forgot-password']

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Get session token from cookie
  const sessionToken = request.cookies.get('session-token')?.value
  const isAuthenticated = !!sessionToken

  // Protect dashboard routes
  const isProtectedRoute = protectedRoutes.some((route) =>
    pathname.startsWith(route)
  )

  if (isProtectedRoute && !isAuthenticated) {
    const url = new URL('/login', request.url)
    url.searchParams.set('callbackUrl', pathname)
    return NextResponse.redirect(url)
  }

  // Redirect to dashboard if already authenticated
  const isAuthRoute = authRoutes.some((route) => pathname.startsWith(route))

  if (isAuthRoute && isAuthenticated) {
    return NextResponse.redirect(new URL('/dashboard', request.url))
  }

  // Add tenant context header if available
  const tenantSlug = request.cookies.get('tenant-slug')?.value
  if (tenantSlug) {
    const headers = new Headers(request.headers)
    headers.set('x-tenant-slug', tenantSlug)
    return NextResponse.next({ headers })
  }

  return NextResponse.next()
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}
