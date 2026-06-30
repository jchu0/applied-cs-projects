// k6 load test for Auth Service
// Run with: k6 run --vus 50 --duration 30s loadtest/auth.js

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const loginErrorRate = new Rate('login_errors');
const loginDuration = new Trend('login_duration');
const tokenRefreshDuration = new Trend('token_refresh_duration');
const logoutDuration = new Trend('logout_duration');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const GATEWAY_URL = __ENV.GATEWAY_URL || 'http://localhost:8000';

// Test configuration
export const options = {
    stages: [
        { duration: '30s', target: 10 },   // Ramp up to 10 users
        { duration: '1m', target: 50 },    // Ramp up to 50 users
        { duration: '2m', target: 50 },    // Stay at 50 users
        { duration: '30s', target: 100 },  // Ramp up to 100 users
        { duration: '1m', target: 100 },   // Stay at 100 users
        { duration: '30s', target: 0 },    // Ramp down to 0
    ],
    thresholds: {
        http_req_duration: ['p(95)<500'],     // 95% of requests should be below 500ms
        http_req_failed: ['rate<0.01'],       // Error rate should be below 1%
        login_errors: ['rate<0.05'],          // Login errors below 5%
        login_duration: ['p(95)<1000'],       // 95% of logins under 1s
        token_refresh_duration: ['p(95)<500'], // 95% of refreshes under 500ms
    },
};

// Test data
const testUsers = [];
for (let i = 0; i < 100; i++) {
    testUsers.push({
        email: `loadtest-${i}@example.com`,
        password: 'LoadTest123!',
        tenant_id: `tenant-${i % 10}`,
    });
}

// Setup: Create test users if they don't exist
export function setup() {
    const createdUsers = [];

    for (const user of testUsers.slice(0, 10)) {
        const registerPayload = JSON.stringify({
            email: user.email,
            password: user.password,
            first_name: 'Load',
            last_name: 'Test',
            tenant_id: user.tenant_id,
        });

        const res = http.post(`${BASE_URL}/api/v1/users`, registerPayload, {
            headers: { 'Content-Type': 'application/json' },
        });

        if (res.status === 201 || res.status === 409) {
            createdUsers.push(user);
        }
    }

    return { users: createdUsers };
}

// Main test function
export default function (data) {
    const user = testUsers[randomIntBetween(0, testUsers.length - 1)];

    group('Authentication Flow', function () {
        // Login
        let loginStart = Date.now();
        const loginPayload = JSON.stringify({
            email: user.email,
            password: user.password,
        });

        const loginRes = http.post(`${GATEWAY_URL}/api/v1/auth/login`, loginPayload, {
            headers: { 'Content-Type': 'application/json' },
        });

        loginDuration.add(Date.now() - loginStart);

        const loginSuccess = check(loginRes, {
            'login status is 200': (r) => r.status === 200,
            'login has access_token': (r) => r.json('access_token') !== undefined,
            'login has refresh_token': (r) => r.json('refresh_token') !== undefined,
        });

        loginErrorRate.add(!loginSuccess);

        if (!loginSuccess) {
            console.error(`Login failed: ${loginRes.status} - ${loginRes.body}`);
            return;
        }

        const accessToken = loginRes.json('access_token');
        const refreshToken = loginRes.json('refresh_token');

        sleep(randomIntBetween(1, 3));

        // Make authenticated request
        group('Authenticated Requests', function () {
            const userRes = http.get(`${GATEWAY_URL}/api/v1/users/me`, {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'Content-Type': 'application/json',
                },
            });

            check(userRes, {
                'get user status is 200': (r) => r.status === 200,
                'user data is valid': (r) => r.json('email') !== undefined,
            });
        });

        sleep(randomIntBetween(1, 2));

        // Refresh token
        group('Token Refresh', function () {
            const refreshStart = Date.now();
            const refreshPayload = JSON.stringify({
                refresh_token: refreshToken,
            });

            const refreshRes = http.post(`${GATEWAY_URL}/api/v1/auth/refresh`, refreshPayload, {
                headers: { 'Content-Type': 'application/json' },
            });

            tokenRefreshDuration.add(Date.now() - refreshStart);

            check(refreshRes, {
                'refresh status is 200': (r) => r.status === 200,
                'refresh has new access_token': (r) => r.json('access_token') !== undefined,
            });
        });

        sleep(randomIntBetween(1, 2));

        // Logout
        group('Logout', function () {
            const logoutStart = Date.now();
            const logoutRes = http.post(`${GATEWAY_URL}/api/v1/auth/logout`, null, {
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                },
            });

            logoutDuration.add(Date.now() - logoutStart);

            check(logoutRes, {
                'logout status is 200 or 204': (r) => r.status === 200 || r.status === 204,
            });
        });
    });

    sleep(randomIntBetween(1, 3));
}

// Teardown
export function teardown(data) {
    console.log('Load test completed');
}
