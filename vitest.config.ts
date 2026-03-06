import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    exclude: ['src/mcp-integration.test.ts'],
    globals: true,
    testTimeout: 10_000,
  },
});
