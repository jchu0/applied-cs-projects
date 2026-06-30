/**
 * Jest configuration for the SaaS Web Platform test suite
 */

module.exports = {
  // Test environment configurations
  projects: [
    {
      displayName: 'frontend',
      testEnvironment: 'jsdom',
      testMatch: ['<rootDir>/tests/frontend/**/*.test.[jt]s?(x)'],
      moduleNameMapper: {
        '^@/(.*)$': '<rootDir>/frontend/$1',
        '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
        '\\.(jpg|jpeg|png|gif|svg)$': '<rootDir>/tests/mocks/fileMock.js',
      },
      setupFilesAfterEnv: ['<rootDir>/tests/frontend/setupTests.ts'],
      transform: {
        '^.+\\.(js|jsx|ts|tsx)$': ['babel-jest', {
          presets: [
            ['@babel/preset-env', { targets: { node: 'current' } }],
            '@babel/preset-react',
            '@babel/preset-typescript',
          ],
        }],
      },
      coverageDirectory: '<rootDir>/coverage/frontend',
      coveragePathIgnorePatterns: [
        '/node_modules/',
        '/tests/',
        '/.next/',
      ],
    },
    {
      displayName: 'backend',
      testEnvironment: 'node',
      testMatch: ['<rootDir>/tests/backend/**/*.test.py'],
      runner: '@jest-runner/python',
      moduleFileExtensions: ['py'],
      coverageDirectory: '<rootDir>/coverage/backend',
      coveragePathIgnorePatterns: [
        '/migrations/',
        '/__pycache__/',
        '/venv/',
      ],
    },
    {
      displayName: 'integration',
      testEnvironment: 'node',
      testMatch: ['<rootDir>/tests/integration/**/*.test.[jt]s'],
      setupFilesAfterEnv: ['<rootDir>/tests/integration/setupTests.js'],
      testTimeout: 30000,
      coverageDirectory: '<rootDir>/coverage/integration',
    },
  ],

  // Coverage configuration
  collectCoverageFrom: [
    'frontend/**/*.{js,jsx,ts,tsx}',
    'backend/**/*.py',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!**/.next/**',
    '!**/migrations/**',
    '!**/coverage/**',
    '!**/dist/**',
    '!**/build/**',
  ],

  coverageReporters: ['text', 'lcov', 'html'],

  coverageThresholds: {
    global: {
      branches: 60,
      functions: 60,
      lines: 60,
      statements: 60,
    },
  },

  // Watch mode configuration
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname',
  ],

  // General configuration
  verbose: true,
  maxWorkers: '50%',
  cacheDirectory: '<rootDir>/.jest-cache',
};