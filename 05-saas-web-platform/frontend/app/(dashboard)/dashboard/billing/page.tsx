'use client';

import { useEffect, useState } from 'react';
import { useApi } from '@/lib/hooks/use-api';
import { useTenant } from '@/lib/hooks/use-tenant';

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
  is_active: boolean;
}

interface Subscription {
  id: string;
  plan: Plan;
  status: string;
  billing_interval: 'monthly' | 'yearly';
  current_period_end: string;
  cancel_at_period_end: boolean;
}

interface Invoice {
  id: string;
  invoice_number: string;
  amount: number;
  status: string;
  invoice_date: string;
  pdf_url: string | null;
}

interface PaymentMethod {
  id: string;
  brand: string;
  last4: string;
  exp_month: number;
  exp_year: number;
  is_default: boolean;
}

// Default plans for static rendering when API is unavailable
const defaultPlans: Plan[] = [
  {
    id: '1',
    name: 'Free',
    slug: 'free',
    description: 'For individuals just getting started',
    price_monthly: 0,
    price_yearly: 0,
    max_users: 3,
    max_storage_gb: 5,
    features: { basic_analytics: true },
    is_active: true,
  },
  {
    id: '2',
    name: 'Pro',
    slug: 'pro',
    description: 'For growing teams',
    price_monthly: 2900,
    price_yearly: 29000,
    max_users: 10,
    max_storage_gb: 50,
    features: { basic_analytics: true, advanced_analytics: true, api_access: true },
    is_active: true,
  },
  {
    id: '3',
    name: 'Enterprise',
    slug: 'enterprise',
    description: 'For large organizations',
    price_monthly: 9900,
    price_yearly: 99000,
    max_users: 100,
    max_storage_gb: 500,
    features: { basic_analytics: true, advanced_analytics: true, api_access: true, sso: true, audit_logs: true },
    is_active: true,
  },
];

const featureLabels: Record<string, string> = {
  basic_analytics: 'Basic Analytics',
  advanced_analytics: 'Advanced Analytics',
  api_access: 'API Access',
  sso: 'Single Sign-On (SSO)',
  audit_logs: 'Audit Logs',
  priority_support: 'Priority Support',
  custom_integrations: 'Custom Integrations',
};

export default function BillingPage() {
  const { get, post, isLoading, error } = useApi();
  const { currentTenant } = useTenant();
  const [plans, setPlans] = useState<Plan[]>(defaultPlans);
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [paymentMethods, setPaymentMethods] = useState<PaymentMethod[]>([]);
  const [loadingPortal, setLoadingPortal] = useState(false);

  useEffect(() => {
    loadPlans();
  }, []);

  useEffect(() => {
    if (currentTenant?.id) {
      loadBillingData();
    }
  }, [currentTenant?.id]);

  const loadPlans = async () => {
    const data = await get('billing/plans');
    if (data && data.length > 0) {
      setPlans(data);
    }
  };

  const loadBillingData = async () => {
    if (!currentTenant?.id) return;

    // Load subscription, invoices, and payment methods in parallel
    const [subData, invoiceData, paymentData] = await Promise.all([
      get(`billing/tenants/${currentTenant.id}/subscription`).catch(() => null),
      get(`billing/tenants/${currentTenant.id}/invoices`).catch(() => []),
      get(`billing/tenants/${currentTenant.id}/payment-methods`).catch(() => []),
    ]);

    if (subData) setSubscription(subData);
    if (invoiceData) setInvoices(invoiceData);
    if (paymentData) setPaymentMethods(paymentData);
  };

  const handleManageBilling = async () => {
    if (!currentTenant?.id) return;
    setLoadingPortal(true);
    try {
      const data = await post(`billing/tenants/${currentTenant.id}/portal`, {
        return_url: window.location.href,
      });
      if (data?.portal_url) {
        window.location.href = data.portal_url;
      }
    } catch (err) {
      console.error('Failed to open billing portal:', err);
    } finally {
      setLoadingPortal(false);
    }
  };

  const handleUpgrade = async (planId: string) => {
    if (!currentTenant?.id) return;
    try {
      const data = await post(`billing/tenants/${currentTenant.id}/checkout`, {
        plan_id: planId,
        billing_interval: 'monthly',
        success_url: `${window.location.origin}/dashboard/billing?success=true`,
        cancel_url: `${window.location.origin}/dashboard/billing?canceled=true`,
      });
      if (data?.checkout_url) {
        window.location.href = data.checkout_url;
      }
    } catch (err) {
      console.error('Failed to start checkout:', err);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount / 100);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const getCurrentPlan = () => {
    if (subscription?.plan) {
      return subscription.plan;
    }
    // Default to free plan if no subscription
    return plans.find(p => p.slug === 'free') || plans[0];
  };

  const currentPlan = getCurrentPlan();

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Billing</h1>
      <p className="mt-1 text-sm text-gray-500">
        Manage your subscription and billing information.
      </p>

      {error && (
        <div className="mt-4 rounded-md bg-red-50 p-4 text-sm text-red-700">
          {error.message}
        </div>
      )}

      {/* Current Plan */}
      <div className="mt-8 rounded-lg bg-white p-6 shadow">
        <h2 className="text-lg font-medium text-gray-900">Current Plan</h2>
        <div className="mt-4 flex items-center justify-between">
          <div>
            <p className="text-3xl font-bold text-gray-900">{currentPlan?.name} Plan</p>
            <p className="mt-1 text-sm text-gray-500">
              {formatCurrency(currentPlan?.price_monthly || 0)}/month
              {subscription?.billing_interval === 'yearly' && ', billed yearly'}
            </p>
            {subscription?.cancel_at_period_end && (
              <p className="mt-1 text-sm text-orange-600">
                Cancels at end of billing period
              </p>
            )}
          </div>
          <button
            onClick={handleManageBilling}
            disabled={loadingPortal || !subscription}
            className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            {loadingPortal ? 'Loading...' : 'Manage subscription'}
          </button>
        </div>
        {subscription?.current_period_end && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">Next billing date</span>
              <span className="font-medium text-gray-900">
                {formatDate(subscription.current_period_end)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Plans */}
      <div className="mt-8">
        <h2 className="text-lg font-medium text-gray-900">Available Plans</h2>
        <div className="mt-4 grid gap-6 lg:grid-cols-3">
          {plans.map((plan) => {
            const isCurrent = plan.id === currentPlan?.id;
            return (
              <div
                key={plan.id}
                className={`rounded-lg border-2 p-6 ${
                  isCurrent
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 bg-white'
                }`}
              >
                <h3 className="text-lg font-medium text-gray-900">{plan.name}</h3>
                <p className="mt-2 text-3xl font-bold text-gray-900">
                  {formatCurrency(plan.price_monthly)}
                  <span className="text-sm font-normal text-gray-500">/month</span>
                </p>
                <p className="mt-2 text-sm text-gray-500">{plan.description}</p>
                <ul className="mt-4 space-y-2">
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckIcon className="mr-2" />
                    Up to {plan.max_users} team members
                  </li>
                  <li className="flex items-center text-sm text-gray-600">
                    <CheckIcon className="mr-2" />
                    {plan.max_storage_gb} GB storage
                  </li>
                  {Object.entries(plan.features).map(([feature, enabled]) => (
                    enabled && (
                      <li key={feature} className="flex items-center text-sm text-gray-600">
                        <CheckIcon className="mr-2" />
                        {featureLabels[feature] || feature}
                      </li>
                    )
                  ))}
                </ul>
                <button
                  onClick={() => !isCurrent && handleUpgrade(plan.id)}
                  disabled={isCurrent || isLoading}
                  className={`mt-6 w-full rounded-md px-4 py-2 text-sm font-medium ${
                    isCurrent
                      ? 'bg-gray-100 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {isCurrent ? 'Current plan' : 'Upgrade'}
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Payment Method */}
      <div className="mt-8 rounded-lg bg-white p-6 shadow">
        <h2 className="text-lg font-medium text-gray-900">Payment Method</h2>
        {paymentMethods.length > 0 ? (
          <div className="mt-4 space-y-3">
            {paymentMethods.map((method) => (
              <div key={method.id} className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="h-8 w-12 rounded bg-gray-200 flex items-center justify-center text-xs font-medium">
                    {method.brand.toUpperCase()}
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-900">
                      {method.brand} ending in {method.last4}
                      {method.is_default && (
                        <span className="ml-2 text-xs text-green-600">(Default)</span>
                      )}
                    </p>
                    <p className="text-sm text-gray-500">
                      Expires {method.exp_month}/{method.exp_year}
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleManageBilling}
                  className="text-sm font-medium text-blue-600 hover:text-blue-500"
                >
                  Update
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-4">
            <p className="text-sm text-gray-500">No payment method on file.</p>
            <button
              onClick={handleManageBilling}
              className="mt-2 text-sm font-medium text-blue-600 hover:text-blue-500"
            >
              Add payment method
            </button>
          </div>
        )}
      </div>

      {/* Billing History */}
      <div className="mt-8 rounded-lg bg-white p-6 shadow">
        <h2 className="text-lg font-medium text-gray-900">Billing History</h2>
        <div className="mt-4 overflow-hidden">
          {invoices.length > 0 ? (
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Date
                  </th>
                  <th className="py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Invoice
                  </th>
                  <th className="py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Amount
                  </th>
                  <th className="py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    Status
                  </th>
                  <th className="relative py-3">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {invoices.map((invoice) => (
                  <tr key={invoice.id}>
                    <td className="whitespace-nowrap py-4 text-sm text-gray-900">
                      {formatDate(invoice.invoice_date)}
                    </td>
                    <td className="whitespace-nowrap py-4 text-sm text-gray-500">
                      {invoice.invoice_number}
                    </td>
                    <td className="whitespace-nowrap py-4 text-sm text-gray-900">
                      {formatCurrency(invoice.amount)}
                    </td>
                    <td className="whitespace-nowrap py-4">
                      <span
                        className={`inline-flex rounded-full px-2 text-xs font-semibold leading-5 ${
                          invoice.status === 'paid'
                            ? 'bg-green-100 text-green-800'
                            : invoice.status === 'open'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {invoice.status}
                      </span>
                    </td>
                    <td className="whitespace-nowrap py-4 text-right text-sm">
                      {invoice.pdf_url && (
                        <a
                          href={invoice.pdf_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-medium text-blue-600 hover:text-blue-500"
                        >
                          Download
                        </a>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-gray-500">No invoices yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      className={`h-4 w-4 text-green-500 ${className || ''}`}
      fill="currentColor"
      viewBox="0 0 20 20"
    >
      <path
        fillRule="evenodd"
        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
        clipRule="evenodd"
      />
    </svg>
  );
}
