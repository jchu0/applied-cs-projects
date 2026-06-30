import Link from 'next/link';

const features = [
  {
    category: 'Authentication & Security',
    items: [
      {
        name: 'Multi-factor Authentication',
        description: 'Secure your accounts with TOTP-based two-factor authentication for an extra layer of protection.',
      },
      {
        name: 'OAuth Integration',
        description: 'Enable sign-in with Google, GitHub, and other popular OAuth providers.',
      },
      {
        name: 'Role-based Access Control',
        description: 'Fine-grained permissions system to control who can access what within your organization.',
      },
      {
        name: 'Audit Logs',
        description: 'Track all actions within your organization for compliance and security monitoring.',
      },
    ],
  },
  {
    category: 'Team Collaboration',
    items: [
      {
        name: 'Multi-tenant Architecture',
        description: 'Create multiple organizations and switch between them seamlessly.',
      },
      {
        name: 'Team Invitations',
        description: 'Invite team members via email with customizable role assignments.',
      },
      {
        name: 'Member Management',
        description: 'Manage team members, update roles, and remove access when needed.',
      },
      {
        name: 'Activity Dashboard',
        description: 'See what your team is working on with real-time activity feeds.',
      },
    ],
  },
  {
    category: 'Billing & Subscriptions',
    items: [
      {
        name: 'Stripe Integration',
        description: 'Secure payment processing with Stripe for subscriptions and one-time payments.',
      },
      {
        name: 'Multiple Pricing Tiers',
        description: 'Support for free, pro, and enterprise plans with custom pricing.',
      },
      {
        name: 'Usage-based Billing',
        description: 'Bill customers based on their actual usage with metered billing.',
      },
      {
        name: 'Invoice Management',
        description: 'Automatic invoice generation and access to billing history.',
      },
    ],
  },
  {
    category: 'Developer Experience',
    items: [
      {
        name: 'REST API',
        description: 'Full-featured API for integrating with your existing tools and workflows.',
      },
      {
        name: 'API Key Management',
        description: 'Generate and manage API keys with customizable permissions and expiration.',
      },
      {
        name: 'Webhooks',
        description: 'Receive real-time notifications about events in your application.',
      },
      {
        name: 'SDK Libraries',
        description: 'Official SDKs for Python, JavaScript, and other popular languages.',
      },
    ],
  },
];

export default function FeaturesPage() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <Link href="/" className="font-bold text-xl">
            SaaS Platform
          </Link>
          <nav className="flex items-center gap-4">
            <Link href="/features" className="text-sm font-medium">
              Features
            </Link>
            <Link href="/pricing" className="text-sm text-muted-foreground hover:text-foreground">
              Pricing
            </Link>
            <Link href="/login" className="text-sm">
              Log in
            </Link>
            <Link
              href="/register"
              className="text-sm bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90"
            >
              Get Started
            </Link>
          </nav>
        </div>
      </header>

      <main className="flex-1 py-16">
        <div className="container mx-auto px-4">
          {/* Hero */}
          <div className="text-center mb-16">
            <h1 className="text-4xl font-bold tracking-tight mb-4">
              Everything you need to build your SaaS
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              A comprehensive platform with all the features you need to launch, grow, and scale your software business.
            </p>
          </div>

          {/* Feature Categories */}
          <div className="space-y-20">
            {features.map((category) => (
              <div key={category.category}>
                <h2 className="text-2xl font-bold mb-8 text-center">{category.category}</h2>
                <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
                  {category.items.map((feature) => (
                    <div key={feature.name} className="border rounded-lg p-6">
                      <div className="flex items-start gap-4">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                          <CheckIcon className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                          <h3 className="font-semibold mb-2">{feature.name}</h3>
                          <p className="text-sm text-muted-foreground">{feature.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* CTA */}
          <div className="mt-20 text-center">
            <h2 className="text-2xl font-bold mb-4">Ready to get started?</h2>
            <p className="text-muted-foreground mb-8">
              Start your free trial today. No credit card required.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Link
                href="/register"
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
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>&copy; 2024 SaaS Platform. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className={className}
    >
      <path
        fillRule="evenodd"
        d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"
        clipRule="evenodd"
      />
    </svg>
  );
}
