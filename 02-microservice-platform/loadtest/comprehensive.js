// Comprehensive k6 load test for all microservices
// Run with: k6 run --vus 100 --duration 5m loadtest/comprehensive.js

import http from 'k6/http';
import { check, sleep, group, fail } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomIntBetween, randomItem, randomString } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('error_rate');
const authLatency = new Trend('auth_latency');
const userLatency = new Trend('user_latency');
const billingLatency = new Trend('billing_latency');
const gatewayLatency = new Trend('gateway_latency');
const requestCounter = new Counter('total_requests');

// Configuration
const GATEWAY_URL = __ENV.GATEWAY_URL || 'http://localhost:8000';
const GRPC_USER_URL = __ENV.GRPC_USER_URL || 'localhost:50051';
const GRPC_AUTH_URL = __ENV.GRPC_AUTH_URL || 'localhost:50052';
const GRPC_BILLING_URL = __ENV.GRPC_BILLING_URL || 'localhost:50053';

// Test configuration
export const options = {
    scenarios: {
        // Normal load
        steady_state: {
            executor: 'constant-arrival-rate',
            rate: 100,
            timeUnit: '1s',
            duration: '2m',
            preAllocatedVUs: 50,
            maxVUs: 200,
        },
        // Stress test
        stress_test: {
            executor: 'ramping-arrival-rate',
            startRate: 50,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 500,
            stages: [
                { duration: '30s', target: 100 },
                { duration: '1m', target: 200 },
                { duration: '30s', target: 300 },
                { duration: '30s', target: 100 },
            ],
            startTime: '2m',
        },
        // Soak test
        soak_test: {
            executor: 'constant-vus',
            vus: 30,
            duration: '5m',
            startTime: '5m',
        },
    },
    thresholds: {
        http_req_duration: ['p(90)<500', 'p(95)<1000'],
        http_req_failed: ['rate<0.01'],
        error_rate: ['rate<0.05'],
        auth_latency: ['p(95)<800'],
        user_latency: ['p(95)<400'],
        billing_latency: ['p(95)<1000'],
        gateway_latency: ['p(95)<100'],
    },
};

// Shared test data
const testData = {
    tenants: Array.from({ length: 10 }, (_, i) => `tenant-${i}`),
    plans: ['free', 'basic', 'pro', 'enterprise'],
    actions: ['create', 'read', 'update', 'delete'],
};

// Token cache
let tokens = {};

// Setup
export function setup() {
    // Create test users and get tokens
    const setupTokens = {};

    for (let i = 0; i < 5; i++) {
        const email = `loadtest-user-${i}@example.com`;

        // Try to register
        http.post(`${GATEWAY_URL}/api/v1/users`, JSON.stringify({
            email: email,
            password: 'LoadTest123!',
            first_name: 'Load',
            last_name: `Test${i}`,
            tenant_id: testData.tenants[i % testData.tenants.length],
        }), {
            headers: { 'Content-Type': 'application/json' },
        });

        // Login to get token
        const loginRes = http.post(`${GATEWAY_URL}/api/v1/auth/login`, JSON.stringify({
            email: email,
            password: 'LoadTest123!',
        }), {
            headers: { 'Content-Type': 'application/json' },
        });

        if (loginRes.status === 200) {
            setupTokens[email] = loginRes.json('access_token');
        }
    }

    return { tokens: setupTokens, tenants: testData.tenants };
}

// Main test function
export default function (data) {
    const tenantId = randomItem(data.tenants);
    const userEmail = `loadtest-user-${randomIntBetween(0, 4)}@example.com`;
    const accessToken = data.tokens[userEmail] || '';

    const authHeaders = {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
        'X-Tenant-ID': tenantId,
        'X-Request-ID': `req-${__VU}-${__ITER}-${Date.now()}`,
    };

    // Choose a random test scenario
    const scenario = randomIntBetween(1, 10);

    if (scenario <= 4) {
        // 40% - User operations
        testUserOperations(authHeaders, tenantId);
    } else if (scenario <= 7) {
        // 30% - Authentication operations
        testAuthOperations(tenantId);
    } else if (scenario <= 9) {
        // 20% - Billing operations
        testBillingOperations(authHeaders, tenantId);
    } else {
        // 10% - Mixed operations
        testMixedOperations(authHeaders, tenantId);
    }

    requestCounter.add(1);
    sleep(randomIntBetween(1, 3));
}

function testUserOperations(headers, tenantId) {
    group('User Service', function () {
        // List users
        group('List Users', function () {
            const startTime = Date.now();
            const res = http.get(`${GATEWAY_URL}/api/v1/users?tenant_id=${tenantId}&limit=10`, {
                headers: headers,
            });

            userLatency.add(Date.now() - startTime);

            const success = check(res, {
                'list users status is 200': (r) => r.status === 200,
            });
            errorRate.add(!success);
        });

        // Get current user
        group('Get Current User', function () {
            const startTime = Date.now();
            const res = http.get(`${GATEWAY_URL}/api/v1/users/me`, {
                headers: headers,
            });

            userLatency.add(Date.now() - startTime);

            check(res, {
                'get me status is 200 or 401': (r) => r.status === 200 || r.status === 401,
            });
        });

        // Update user profile
        group('Update Profile', function () {
            const startTime = Date.now();
            const res = http.patch(`${GATEWAY_URL}/api/v1/users/me`, JSON.stringify({
                first_name: `Updated${Date.now()}`,
            }), {
                headers: headers,
            });

            userLatency.add(Date.now() - startTime);

            check(res, {
                'update profile status is 200 or 401': (r) => r.status === 200 || r.status === 401,
            });
        });
    });
}

function testAuthOperations(tenantId) {
    group('Auth Service', function () {
        const email = `temp-${randomString(8)}@loadtest.com`;
        const password = 'TempPass123!';

        // Register
        group('Register', function () {
            const startTime = Date.now();
            const res = http.post(`${GATEWAY_URL}/api/v1/auth/register`, JSON.stringify({
                email: email,
                password: password,
                tenant_id: tenantId,
            }), {
                headers: { 'Content-Type': 'application/json' },
            });

            authLatency.add(Date.now() - startTime);

            check(res, {
                'register status is 201 or 409': (r) => r.status === 201 || r.status === 409,
            });
        });

        // Login
        let accessToken = '';
        let refreshToken = '';
        group('Login', function () {
            const startTime = Date.now();
            const res = http.post(`${GATEWAY_URL}/api/v1/auth/login`, JSON.stringify({
                email: email,
                password: password,
            }), {
                headers: { 'Content-Type': 'application/json' },
            });

            authLatency.add(Date.now() - startTime);

            const success = check(res, {
                'login status is 200 or 401': (r) => r.status === 200 || r.status === 401,
            });

            if (res.status === 200) {
                accessToken = res.json('access_token') || '';
                refreshToken = res.json('refresh_token') || '';
            }

            errorRate.add(!success && res.status !== 401);
        });

        if (accessToken) {
            // Refresh token
            group('Refresh Token', function () {
                const startTime = Date.now();
                const res = http.post(`${GATEWAY_URL}/api/v1/auth/refresh`, JSON.stringify({
                    refresh_token: refreshToken,
                }), {
                    headers: { 'Content-Type': 'application/json' },
                });

                authLatency.add(Date.now() - startTime);

                check(res, {
                    'refresh status is 200': (r) => r.status === 200,
                });
            });

            // Logout
            group('Logout', function () {
                const startTime = Date.now();
                const res = http.post(`${GATEWAY_URL}/api/v1/auth/logout`, null, {
                    headers: {
                        'Authorization': `Bearer ${accessToken}`,
                    },
                });

                authLatency.add(Date.now() - startTime);

                check(res, {
                    'logout status is 200 or 204': (r) => r.status === 200 || r.status === 204,
                });
            });
        }
    });
}

function testBillingOperations(headers, tenantId) {
    group('Billing Service', function () {
        const customerId = `cust-${__VU}-${Date.now()}`;

        // Create customer
        group('Create Customer', function () {
            const startTime = Date.now();
            const res = http.post(`${GATEWAY_URL}/api/v1/billing/customers`, JSON.stringify({
                email: `${customerId}@loadtest.com`,
                name: `Customer ${customerId}`,
                tenant_id: tenantId,
            }), {
                headers: headers,
            });

            billingLatency.add(Date.now() - startTime);

            check(res, {
                'create customer status is 201 or 200': (r) => r.status === 201 || r.status === 200,
            });
        });

        // Get pricing
        group('Get Pricing', function () {
            const startTime = Date.now();
            const res = http.get(`${GATEWAY_URL}/api/v1/billing/pricing`, {
                headers: headers,
            });

            billingLatency.add(Date.now() - startTime);

            check(res, {
                'get pricing status is 200': (r) => r.status === 200,
            });
        });

        // List invoices
        group('List Invoices', function () {
            const startTime = Date.now();
            const res = http.get(`${GATEWAY_URL}/api/v1/billing/invoices?tenant_id=${tenantId}&limit=10`, {
                headers: headers,
            });

            billingLatency.add(Date.now() - startTime);

            check(res, {
                'list invoices status is 200': (r) => r.status === 200,
            });
        });
    });
}

function testMixedOperations(headers, tenantId) {
    group('Mixed Operations', function () {
        // Simulate typical user flow

        // 1. Get user info
        http.get(`${GATEWAY_URL}/api/v1/users/me`, { headers });

        sleep(0.5);

        // 2. Check subscription status
        http.get(`${GATEWAY_URL}/api/v1/billing/subscriptions?tenant_id=${tenantId}`, { headers });

        sleep(0.5);

        // 3. Update preferences
        http.patch(`${GATEWAY_URL}/api/v1/users/me/preferences`, JSON.stringify({
            notifications: true,
            theme: 'dark',
        }), { headers });

        // 4. Verify session
        http.get(`${GATEWAY_URL}/api/v1/auth/verify`, { headers });
    });
}

// Gateway health check
export function handleSummary(data) {
    const healthRes = http.get(`${GATEWAY_URL}/health`);

    return {
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
        'loadtest/results.json': JSON.stringify(data),
    };
}

function textSummary(data, options) {
    // Simple text summary
    let summary = '\n=== Load Test Summary ===\n\n';

    summary += `Total Requests: ${data.metrics.http_reqs?.values?.count || 0}\n`;
    summary += `Requests/sec: ${data.metrics.http_reqs?.values?.rate?.toFixed(2) || 0}\n`;
    summary += `Error Rate: ${(data.metrics.http_req_failed?.values?.rate * 100 || 0).toFixed(2)}%\n`;
    summary += `Avg Response Time: ${(data.metrics.http_req_duration?.values?.avg || 0).toFixed(2)}ms\n`;
    summary += `P95 Response Time: ${(data.metrics.http_req_duration?.values?.['p(95)'] || 0).toFixed(2)}ms\n`;

    return summary;
}
