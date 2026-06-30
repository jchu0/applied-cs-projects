// k6 load test for Billing Service
// Run with: k6 run --vus 20 --duration 30s loadtest/billing.js

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend } from 'k6/metrics';
import { randomIntBetween, randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const subscriptionCreateDuration = new Trend('subscription_create_duration');
const subscriptionGetDuration = new Trend('subscription_get_duration');
const invoiceListDuration = new Trend('invoice_list_duration');
const billingErrorRate = new Rate('billing_errors');

// Configuration
const GATEWAY_URL = __ENV.GATEWAY_URL || 'http://localhost:8000';

// Test configuration
export const options = {
    scenarios: {
        // Steady load for subscription management
        subscription_management: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '20s', target: 10 },
                { duration: '1m', target: 20 },
                { duration: '30s', target: 0 },
            ],
            gracefulRampDown: '10s',
        },
        // Spike test for invoice viewing
        invoice_spike: {
            executor: 'ramping-arrival-rate',
            startRate: 10,
            timeUnit: '1s',
            preAllocatedVUs: 50,
            maxVUs: 100,
            stages: [
                { duration: '30s', target: 10 },
                { duration: '10s', target: 50 },   // Spike
                { duration: '30s', target: 10 },
            ],
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<800'],
        http_req_failed: ['rate<0.02'],
        billing_errors: ['rate<0.05'],
        subscription_create_duration: ['p(95)<2000'],
        subscription_get_duration: ['p(95)<300'],
        invoice_list_duration: ['p(95)<500'],
    },
};

// Test data
const plans = ['basic', 'pro', 'enterprise'];
const currencies = ['usd', 'eur', 'gbp'];

// Setup: Get auth token
export function setup() {
    const loginRes = http.post(`${GATEWAY_URL}/api/v1/auth/login`, JSON.stringify({
        email: 'admin@loadtest.com',
        password: 'LoadTest123!',
    }), {
        headers: { 'Content-Type': 'application/json' },
    });

    const accessToken = loginRes.json('access_token') || 'mock-token';
    return { accessToken };
}

export default function (data) {
    const authHeaders = {
        'Authorization': `Bearer ${data.accessToken}`,
        'Content-Type': 'application/json',
    };

    const customerId = `cust-${__VU}-${__ITER}`;
    const tenantId = `tenant-${__VU % 10}`;

    group('Subscription Lifecycle', function () {
        // Create subscription
        group('Create Subscription', function () {
            const startTime = Date.now();
            const payload = JSON.stringify({
                customer_id: customerId,
                tenant_id: tenantId,
                plan: randomItem(plans),
                currency: randomItem(currencies),
                billing_cycle: 'monthly',
            });

            const createRes = http.post(`${GATEWAY_URL}/api/v1/billing/subscriptions`, payload, {
                headers: authHeaders,
            });

            subscriptionCreateDuration.add(Date.now() - startTime);

            const success = check(createRes, {
                'create subscription status is 201 or 200': (r) => r.status === 201 || r.status === 200,
                'subscription has id': (r) => r.json('id') !== undefined || r.json('subscription_id') !== undefined,
            });

            billingErrorRate.add(!success);

            if (!success) {
                return;
            }

            const subscriptionId = createRes.json('id') || createRes.json('subscription_id');

            sleep(randomIntBetween(1, 2));

            // Get subscription
            group('Get Subscription', function () {
                const getStart = Date.now();
                const getRes = http.get(`${GATEWAY_URL}/api/v1/billing/subscriptions/${subscriptionId}`, {
                    headers: authHeaders,
                });

                subscriptionGetDuration.add(Date.now() - getStart);

                check(getRes, {
                    'get subscription status is 200': (r) => r.status === 200,
                    'subscription data is valid': (r) => r.json('customer_id') !== undefined,
                });
            });

            sleep(1);

            // Update subscription
            group('Update Subscription', function () {
                const updatePayload = JSON.stringify({
                    plan: randomItem(plans),
                });

                const updateRes = http.put(
                    `${GATEWAY_URL}/api/v1/billing/subscriptions/${subscriptionId}`,
                    updatePayload,
                    { headers: authHeaders }
                );

                check(updateRes, {
                    'update subscription status is 200': (r) => r.status === 200,
                });
            });
        });
    });

    group('Invoice Operations', function () {
        // List invoices
        group('List Invoices', function () {
            const startTime = Date.now();
            const listRes = http.get(`${GATEWAY_URL}/api/v1/billing/invoices?customer_id=${customerId}&limit=10`, {
                headers: authHeaders,
            });

            invoiceListDuration.add(Date.now() - startTime);

            check(listRes, {
                'list invoices status is 200': (r) => r.status === 200,
                'invoices response is array or object': (r) => {
                    const body = r.json();
                    return Array.isArray(body) || (typeof body === 'object' && body !== null);
                },
            });
        });
    });

    group('Payment Methods', function () {
        // Add payment method
        const paymentPayload = JSON.stringify({
            customer_id: customerId,
            type: 'card',
            card_token: `tok_${Date.now()}`,
        });

        const addRes = http.post(`${GATEWAY_URL}/api/v1/billing/payment-methods`, paymentPayload, {
            headers: authHeaders,
        });

        check(addRes, {
            'add payment method status is 201 or 200': (r) => r.status === 201 || r.status === 200,
        });

        // List payment methods
        const listRes = http.get(`${GATEWAY_URL}/api/v1/billing/payment-methods?customer_id=${customerId}`, {
            headers: authHeaders,
        });

        check(listRes, {
            'list payment methods status is 200': (r) => r.status === 200,
        });
    });

    sleep(randomIntBetween(1, 3));
}

export function teardown(data) {
    console.log('Billing load test completed');
}
