'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';

interface Plan {
  id: string;
  name: string;
  slug: string;
  description: string;
  price_monthly: number;
  price_yearly: number;
  max_users: number;
  max_storage_gb: number;
  features: Record<string, boolean>;
}

// Default plans for static rendering when API is unavailable
const defaultPlans: Plan[] = [
  {
    id: '1',
    name: 'Free',
    slug: 'free',
    description: 'Perfect for getting started',
    price_monthly: 0,
    price_yearly: 0,
    max_users: 2,
    max_storage_gb: 1,
    features: {
      basic_analytics: true,
      email_support: true,
    },
  },
  {
    id: '2',
    name: 'Pro',
    slug: 'pro',
    description: 'For growing teams',
    price_monthly: 29,
    price_yearly: 290,
    max_users: 10,
    max_storage_gb: 50,
    features: {
      basic_analytics: true,
      advanced_analytics: true,
      email_support: true,
      priority_support: true,
      api_access: true,
    },
  },
  {
    id: '3',
    name: 'Enterprise',
    slug: 'enterprise',
    description: 'For large organizations',
    price_monthly: 99,
    price_yearly: 990,
    max_users: 100,
    max_storage_gb: 500,
    features: {
      basic_analytics: true,
      advanced_analytics: true,
      custom_reports: true,
      email_support: true,
      priority_support: true,
      dedicated_support: true,
      api_access: true,
      sso: true,
      audit_logs: true,
      custom_integrations: true,
    },
  },
];

const featureLabels: Record<string, string> = {
  basic_analytics: 'Basic Analytics',
  advanced_analytics: 'Advanced Analytics',
  custom_reports: 'Custom Reports',
  email_support: 'Email Support',
  priority_support: 'Priority Support',
  dedicated_support: 'Dedicated Support Manager',
  api_access: 'API Access',
  sso: 'Single Sign-On (SSO)',
  audit_logs: 'Audit Logs',
  custom_integrations: 'Custom Integrations',
};

export default function PricingPage() {
  const [plans, setPlans] = useState<Plan[]>(defaultPlans);
  const [billingInterval, setBillingInterval] = useState<'monthly' | 'yearly'>('monthly');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchPlans() {
      try {
        const response = await fetch('/api/billing/plans');
        if (response.ok) {
          const data = await response.json();
          if (data && data.length > 0) {
            setPlans(data);
          }
        }
      } catch (error) {
        // Use default plans if API is unavailable
        console.log('Using default plans');
      } finally {
        setLoading(false);
      }
    }
    fetchPlans();
  }, []);

  const getPrice = (plan: Plan) => {
    return billingInterval === 'monthly' ? plan.price_monthly : plan.price_yearly;
  };

  const getPriceLabel = () => {
    return billingInterval === 'monthly' ? '/month' : '/year';
  };

  const getSavings = (plan: Plan) => {
    if (plan.price_monthly === 0) return 0;
    const yearlyMonthly = plan.price_monthly * 12;
    return Math.round((1 - plan.price_yearly / yearlyMonthly) * 100);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <Link href="/" className="font-bold text-xl">
            SaaS Platform
          </Link>
          <nav className="flex items-center gap-4">
            <Link href="/features" className="text-sm text-muted-foreground hover:text-foreground">
              Features
            </Link>
            <Link href="/pricing" className="text-sm font-medium">
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
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold tracking-tight mb-4">
              Simple, transparent pricing
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Choose the plan that&apos;s right for your team. All plans include a 14-day free trial.
            </p>
          </div>

          {/* Billing Toggle */}
          <div className="flex justify-center mb-12">
            <div className="inline-flex items-center gap-4 p-1 bg-muted rounded-lg">
              <button
                onClick={() => setBillingInterval('monthly')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  billingInterval === 'monthly'
                    ? 'bg-background shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Monthly
              </button>
              <button
                onClick={() => setBillingInterval('yearly')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  billingInterval === 'yearly'
                    ? 'bg-background shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Yearly
                <span className="ml-2 text-xs text-green-600 font-normal">Save up to 20%</span>
              </button>
            </div>
          </div>

          {/* Pricing Cards */}
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {loading ? (
              // Loading skeleton
              [...Array(3)].map((_, i) => (
                <div key={i} className="border rounded-lg p-8 animate-pulse">
                  <div className="h-6 bg-muted rounded w-24 mb-4"></div>
                  <div className="h-10 bg-muted rounded w-32 mb-4"></div>
                  <div className="h-4 bg-muted rounded w-full mb-8"></div>
                  <div className="space-y-3">
                    {[...Array(5)].map((_, j) => (
                      <div key={j} className="h-4 bg-muted rounded w-3/4"></div>
                    ))}
                  </div>
                </div>
              ))
            ) : (
              plans.map((plan, index) => {
                const isPopular = index === 1;
                return (
                  <div
                    key={plan.id}
                    className={`border rounded-lg p-8 relative ${
                      isPopular ? 'border-primary shadow-lg scale-105' : ''
                    }`}
                  >
                    {isPopular && (
                      <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                        <span className="bg-primary text-primary-foreground text-xs font-medium px-3 py-1 rounded-full">
                          Most Popular
                        </span>
                      </div>
                    )}

                    <div className="mb-6">
                      <h3 className="text-xl font-semibold mb-2">{plan.name}</h3>
                      <p className="text-sm text-muted-foreground">{plan.description}</p>
                    </div>

                    <div className="mb-6">
                      <div className="flex items-baseline gap-1">
                        <span className="text-4xl font-bold">
                          ${getPrice(plan)}
                        </span>
                        <span className="text-muted-foreground">{getPriceLabel()}</span>
                      </div>
                      {billingInterval === 'yearly' && getSavings(plan) > 0 && (
                        <p className="text-sm text-green-600 mt-1">
                          Save {getSavings(plan)}% with yearly billing
                        </p>
                      )}
                    </div>

                    <Link
                      href={`/register?plan=${plan.slug}`}
                      className={`block text-center py-3 rounded-md font-medium transition-colors ${
                        isPopular
                          ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                          : 'border hover:bg-muted'
                      }`}
                    >
                      {plan.price_monthly === 0 ? 'Get Started Free' : 'Start Free Trial'}
                    </Link>

                    <div className="mt-8 pt-6 border-t">
                      <p className="text-sm font-medium mb-4">What&apos;s included:</p>
                      <ul className="space-y-3">
                        <li className="flex items-start gap-3 text-sm">
                          <CheckIcon className="w-5 h-5 text-green-500 flex-shrink-0" />
                          <span>Up to {plan.max_users} team members</span>
                        </li>
                        <li className="flex items-start gap-3 text-sm">
                          <CheckIcon className="w-5 h-5 text-green-500 flex-shrink-0" />
                          <span>{plan.max_storage_gb} GB storage</span>
                        </li>
                        {Object.entries(plan.features).map(([feature, enabled]) => (
                          enabled && (
                            <li key={feature} className="flex items-start gap-3 text-sm">
                              <CheckIcon className="w-5 h-5 text-green-500 flex-shrink-0" />
                              <span>{featureLabels[feature] || feature}</span>
                            </li>
                          )
                        ))}
                      </ul>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* FAQ Section */}
          <div className="mt-24 max-w-3xl mx-auto">
            <h2 className="text-2xl font-bold text-center mb-8">Frequently asked questions</h2>
            <div className="space-y-6">
              <div className="border-b pb-6">
                <h3 className="font-medium mb-2">Can I switch plans later?</h3>
                <p className="text-muted-foreground text-sm">
                  Yes, you can upgrade or downgrade your plan at any time. Changes take effect at the start of your next billing cycle.
                </p>
              </div>
              <div className="border-b pb-6">
                <h3 className="font-medium mb-2">What payment methods do you accept?</h3>
                <p className="text-muted-foreground text-sm">
                  We accept all major credit cards (Visa, Mastercard, American Express) through our secure payment provider Stripe.
                </p>
              </div>
              <div className="border-b pb-6">
                <h3 className="font-medium mb-2">Is there a free trial?</h3>
                <p className="text-muted-foreground text-sm">
                  Yes! All paid plans come with a 14-day free trial. No credit card required to start.
                </p>
              </div>
              <div className="border-b pb-6">
                <h3 className="font-medium mb-2">Can I cancel anytime?</h3>
                <p className="text-muted-foreground text-sm">
                  Absolutely. You can cancel your subscription at any time with no cancellation fees. Your access continues until the end of your billing period.
                </p>
              </div>
              <div>
                <h3 className="font-medium mb-2">Do you offer discounts for non-profits?</h3>
                <p className="text-muted-foreground text-sm">
                  Yes, we offer special pricing for non-profit organizations and educational institutions. Contact us for more details.
                </p>
              </div>
            </div>
          </div>

          {/* CTA Section */}
          <div className="mt-24 text-center">
            <h2 className="text-2xl font-bold mb-4">Still have questions?</h2>
            <p className="text-muted-foreground mb-6">
              Our team is here to help. Get in touch and we&apos;ll get back to you as soon as possible.
            </p>
            <Link
              href="/contact"
              className="inline-block border px-6 py-3 rounded-md font-medium hover:bg-muted transition-colors"
            >
              Contact Sales
            </Link>
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
