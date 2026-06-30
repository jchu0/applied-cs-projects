import Link from 'next/link'

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="font-bold text-xl">SaaS Platform</div>
          <nav className="flex items-center gap-4">
            <Link href="/features" className="text-sm text-muted-foreground hover:text-foreground">
              Features
            </Link>
            <Link href="/pricing" className="text-sm text-muted-foreground hover:text-foreground">
              Pricing
            </Link>
            <Link href="/login" className="text-sm">
              Log in
            </Link>
            <Link
              href="/signup"
              className="text-sm bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90"
            >
              Get Started
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <main className="flex-1">
        <section className="container mx-auto px-4 py-24 text-center">
          <h1 className="text-5xl font-bold tracking-tight mb-6">
            Build your SaaS faster
          </h1>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            A complete platform for building and scaling your software business.
            Authentication, billing, teams, and more—all included.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Link
              href="/signup"
              className="bg-primary text-primary-foreground px-8 py-3 rounded-md font-medium hover:bg-primary/90"
            >
              Start free trial
            </Link>
            <Link
              href="/demo"
              className="border px-8 py-3 rounded-md font-medium hover:bg-muted"
            >
              Book a demo
            </Link>
          </div>
        </section>

        {/* Features */}
        <section className="container mx-auto px-4 py-16">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="p-6 border rounded-lg">
              <h3 className="font-semibold mb-2">Authentication</h3>
              <p className="text-sm text-muted-foreground">
                Secure authentication with email, OAuth, and MFA support.
              </p>
            </div>
            <div className="p-6 border rounded-lg">
              <h3 className="font-semibold mb-2">Team Management</h3>
              <p className="text-sm text-muted-foreground">
                Invite team members, manage roles, and control permissions.
              </p>
            </div>
            <div className="p-6 border rounded-lg">
              <h3 className="font-semibold mb-2">Subscription Billing</h3>
              <p className="text-sm text-muted-foreground">
                Integrated Stripe billing with multiple pricing tiers.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>&copy; 2024 SaaS Platform. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}
