/**
 * End-to-end integration tests for the SaaS Web Platform
 */

const request = require('supertest');
const puppeteer = require('puppeteer');
const { MongoMemoryServer } = require('mongodb-memory-server');
const { Client } = require('pg');
const Redis = require('ioredis-mock');

describe('E2E: Complete User Journey', () => {
  let browser;
  let page;
  let apiUrl;
  let frontendUrl;
  let mongoServer;
  let pgClient;
  let redisClient;

  beforeAll(async () => {
    // Setup test databases
    mongoServer = await MongoMemoryServer.create();
    process.env.MONGO_URI = mongoServer.getUri();

    pgClient = new Client({
      connectionString: process.env.TEST_DATABASE_URL,
    });
    await pgClient.connect();

    redisClient = new Redis();

    // Start test servers
    apiUrl = process.env.API_URL || 'http://localhost:8000';
    frontendUrl = process.env.FRONTEND_URL || 'http://localhost:3000';

    // Launch browser
    browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
    page = await browser.newPage();
  });

  afterAll(async () => {
    await browser.close();
    await mongoServer.stop();
    await pgClient.end();
    redisClient.disconnect();
  });

  describe('User Registration and Onboarding', () => {
    test('should complete full registration flow', async () => {
      // Navigate to registration page
      await page.goto(`${frontendUrl}/register`);

      // Fill registration form
      await page.type('input[name="username"]', 'testuser123');
      await page.type('input[name="email"]', 'test123@example.com');
      await page.type('input[name="password"]', 'TestPass123!');
      await page.type('input[name="passwordConfirm"]', 'TestPass123!');

      // Submit form
      await page.click('button[type="submit"]');

      // Wait for redirect to dashboard
      await page.waitForNavigation();
      expect(page.url()).toContain('/dashboard');

      // Verify welcome message
      const welcomeText = await page.$eval('.welcome-message', el => el.textContent);
      expect(welcomeText).toContain('Welcome, testuser123');
    });

    test('should complete onboarding flow', async () => {
      // Continue from dashboard
      await page.goto(`${frontendUrl}/onboarding`);

      // Step 1: Organization setup
      await page.type('input[name="orgName"]', 'Test Company');
      await page.select('select[name="orgSize"]', '10-50');
      await page.click('button[data-step="next"]');

      // Step 2: Choose plan
      await page.click('div[data-plan="pro"]');
      await page.click('button[data-step="next"]');

      // Step 3: Invite team members
      await page.type('input[name="inviteEmail1"]', 'member1@example.com');
      await page.type('input[name="inviteEmail2"]', 'member2@example.com');
      await page.click('button[data-step="complete"]');

      // Verify onboarding completion
      await page.waitForNavigation();
      expect(page.url()).toContain('/dashboard');
    });
  });

  describe('Subscription and Billing', () => {
    let sessionCookie;

    beforeEach(async () => {
      // Login and get session
      const loginResponse = await request(apiUrl)
        .post('/api/auth/login')
        .send({
          email: 'test123@example.com',
          password: 'TestPass123!',
        });

      sessionCookie = loginResponse.headers['set-cookie'];
    });

    test('should upgrade subscription plan', async () => {
      // Get current subscription
      const subResponse = await request(apiUrl)
        .get('/api/subscriptions/current')
        .set('Cookie', sessionCookie);

      const currentSub = subResponse.body;

      // Upgrade to enterprise plan
      const upgradeResponse = await request(apiUrl)
        .post(`/api/subscriptions/${currentSub.id}/upgrade`)
        .set('Cookie', sessionCookie)
        .send({
          plan_id: 'enterprise',
          payment_method: 'pm_card_visa',
        });

      expect(upgradeResponse.status).toBe(200);
      expect(upgradeResponse.body.plan).toBe('enterprise');
    });

    test('should process payment successfully', async () => {
      // Create payment intent
      const intentResponse = await request(apiUrl)
        .post('/api/payments/create-intent')
        .set('Cookie', sessionCookie)
        .send({
          amount: 9999,
          currency: 'usd',
        });

      expect(intentResponse.status).toBe(200);
      expect(intentResponse.body).toHaveProperty('client_secret');

      // Confirm payment (mocked)
      const confirmResponse = await request(apiUrl)
        .post('/api/payments/confirm')
        .set('Cookie', sessionCookie)
        .send({
          payment_intent_id: intentResponse.body.id,
          payment_method: 'pm_card_visa',
        });

      expect(confirmResponse.status).toBe(200);
      expect(confirmResponse.body.status).toBe('succeeded');
    });
  });

  describe('Core Features Usage', () => {
    test('should create and manage projects', async () => {
      // Login
      await page.goto(`${frontendUrl}/login`);
      await page.type('input[name="email"]', 'test123@example.com');
      await page.type('input[name="password"]', 'TestPass123!');
      await page.click('button[type="submit"]');
      await page.waitForNavigation();

      // Navigate to projects
      await page.goto(`${frontendUrl}/projects`);

      // Create new project
      await page.click('button[data-action="new-project"]');
      await page.type('input[name="projectName"]', 'Test Project');
      await page.type('textarea[name="description"]', 'Test project description');
      await page.click('button[type="submit"]');

      // Verify project created
      await page.waitForSelector('[data-project="Test Project"]');
      const projectCard = await page.$('[data-project="Test Project"]');
      expect(projectCard).toBeTruthy();

      // Open project settings
      await page.click('[data-project="Test Project"] [data-action="settings"]');

      // Update project
      await page.type('input[name="projectName"]', ' Updated');
      await page.click('button[data-action="save"]');

      // Verify update
      await page.waitForSelector('[data-project="Test Project Updated"]');
    });

    test('should handle file uploads', async () => {
      // Navigate to files section
      await page.goto(`${frontendUrl}/projects/test-project/files`);

      // Upload file
      const fileInput = await page.$('input[type="file"]');
      await fileInput.uploadFile('./tests/fixtures/test-file.pdf');

      // Wait for upload completion
      await page.waitForSelector('.upload-success');

      // Verify file appears in list
      const fileItem = await page.$('[data-file="test-file.pdf"]');
      expect(fileItem).toBeTruthy();
    });

    test('should manage team members', async () => {
      // Navigate to team settings
      await page.goto(`${frontendUrl}/settings/team`);

      // Invite new member
      await page.click('button[data-action="invite-member"]');
      await page.type('input[name="email"]', 'newmember@example.com');
      await page.select('select[name="role"]', 'editor');
      await page.click('button[type="submit"]');

      // Verify invitation sent
      await page.waitForSelector('.invite-success');
      const inviteMessage = await page.$eval('.invite-success', el => el.textContent);
      expect(inviteMessage).toContain('Invitation sent');
    });
  });

  describe('API Integration', () => {
    let apiKey;

    beforeAll(async () => {
      // Generate API key
      const response = await request(apiUrl)
        .post('/api/keys/generate')
        .set('Cookie', sessionCookie)
        .send({ name: 'Test API Key' });

      apiKey = response.body.key;
    });

    test('should authenticate with API key', async () => {
      const response = await request(apiUrl)
        .get('/api/v1/user')
        .set('X-API-Key', apiKey);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('email', 'test123@example.com');
    });

    test('should handle rate limiting', async () => {
      // Make multiple rapid requests
      const requests = Array(101).fill().map(() =>
        request(apiUrl)
          .get('/api/v1/projects')
          .set('X-API-Key', apiKey)
      );

      const responses = await Promise.all(requests);
      const rateLimitedResponse = responses.find(r => r.status === 429);

      expect(rateLimitedResponse).toBeTruthy();
      expect(rateLimitedResponse.body).toHaveProperty('error', 'Rate limit exceeded');
    });

    test('should handle webhooks', async () => {
      // Register webhook
      const webhookResponse = await request(apiUrl)
        .post('/api/webhooks')
        .set('X-API-Key', apiKey)
        .send({
          url: 'https://example.com/webhook',
          events: ['project.created', 'project.updated'],
        });

      expect(webhookResponse.status).toBe(201);
      const webhookId = webhookResponse.body.id;

      // Trigger webhook event
      const projectResponse = await request(apiUrl)
        .post('/api/v1/projects')
        .set('X-API-Key', apiKey)
        .send({
          name: 'Webhook Test Project',
        });

      // Verify webhook was called (check webhook logs)
      const logsResponse = await request(apiUrl)
        .get(`/api/webhooks/${webhookId}/logs`)
        .set('X-API-Key', apiKey);

      expect(logsResponse.body[0].event).toBe('project.created');
      expect(logsResponse.body[0].status).toBe('delivered');
    });
  });

  describe('Performance and Security', () => {
    test('should handle concurrent users', async () => {
      // Simulate multiple concurrent users
      const browsers = await Promise.all(
        Array(10).fill().map(() => puppeteer.launch({ headless: true }))
      );

      const pages = await Promise.all(
        browsers.map(b => b.newPage())
      );

      // All users navigate simultaneously
      await Promise.all(
        pages.map(p => p.goto(`${frontendUrl}/dashboard`))
      );

      // Verify all loaded successfully
      const loadTimes = await Promise.all(
        pages.map(p => p.evaluate(() => performance.timing.loadEventEnd - performance.timing.navigationStart))
      );

      // Check that all pages loaded within acceptable time (5 seconds)
      loadTimes.forEach(time => {
        expect(time).toBeLessThan(5000);
      });

      // Cleanup
      await Promise.all(browsers.map(b => b.close()));
    });

    test('should enforce CORS policies', async () => {
      // Try cross-origin request
      const response = await request(apiUrl)
        .get('/api/v1/user')
        .set('Origin', 'https://malicious-site.com');

      expect(response.status).toBe(403);
      expect(response.body).toHaveProperty('error', 'CORS policy violation');
    });

    test('should handle XSS attempts', async () => {
      // Try injecting script
      await page.goto(`${frontendUrl}/projects`);

      const maliciousInput = '<script>alert("XSS")</script>';
      await page.type('input[name="projectName"]', maliciousInput);
      await page.click('button[type="submit"]');

      // Verify script is escaped
      await page.waitForSelector('[data-project]');
      const projectHTML = await page.$eval('[data-project]', el => el.innerHTML);
      expect(projectHTML).not.toContain('<script>');
      expect(projectHTML).toContain('&lt;script&gt;');
    });

    test('should enforce authentication', async () => {
      // Try accessing protected route without auth
      const response = await request(apiUrl)
        .get('/api/admin/users');

      expect(response.status).toBe(401);
      expect(response.body).toHaveProperty('error', 'Authentication required');
    });
  });

  describe('Error Handling and Recovery', () => {
    test('should handle network failures gracefully', async () => {
      // Simulate offline mode
      await page.setOfflineMode(true);

      await page.goto(`${frontendUrl}/dashboard`);

      // Should show offline message
      await page.waitForSelector('.offline-banner');
      const offlineText = await page.$eval('.offline-banner', el => el.textContent);
      expect(offlineText).toContain('You are offline');

      // Resume online
      await page.setOfflineMode(false);

      // Should auto-reconnect
      await page.waitForSelector('.online-banner');
    });

    test('should handle server errors', async () => {
      // Mock server error
      const response = await request(apiUrl)
        .post('/api/trigger-500')
        .send({});

      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error');
      expect(response.body).toHaveProperty('request_id');
    });

    test('should recover from database failures', async () => {
      // Simulate database connection loss
      await pgClient.end();

      // Try database operation
      const response = await request(apiUrl)
        .get('/api/v1/projects')
        .set('X-API-Key', apiKey);

      expect(response.status).toBe(503);
      expect(response.body).toHaveProperty('error', 'Service temporarily unavailable');

      // Reconnect database
      await pgClient.connect();

      // Retry operation
      const retryResponse = await request(apiUrl)
        .get('/api/v1/projects')
        .set('X-API-Key', apiKey);

      expect(retryResponse.status).toBe(200);
    });
  });

  describe('Data Export and Compliance', () => {
    test('should export user data (GDPR)', async () => {
      const response = await request(apiUrl)
        .post('/api/privacy/export')
        .set('Cookie', sessionCookie);

      expect(response.status).toBe(202);
      expect(response.body).toHaveProperty('export_id');

      // Check export status
      const statusResponse = await request(apiUrl)
        .get(`/api/privacy/export/${response.body.export_id}`)
        .set('Cookie', sessionCookie);

      expect(statusResponse.body.status).toBeOneOf(['pending', 'processing', 'completed']);
    });

    test('should delete user account', async () => {
      // Request account deletion
      const deleteResponse = await request(apiUrl)
        .post('/api/account/delete')
        .set('Cookie', sessionCookie)
        .send({
          password: 'TestPass123!',
          confirmation: 'DELETE',
        });

      expect(deleteResponse.status).toBe(200);
      expect(deleteResponse.body).toHaveProperty('scheduled_deletion_date');

      // Verify account is marked for deletion
      const accountResponse = await request(apiUrl)
        .get('/api/account')
        .set('Cookie', sessionCookie);

      expect(accountResponse.body.status).toBe('pending_deletion');
    });
  });
});