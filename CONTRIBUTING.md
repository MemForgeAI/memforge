# Contributing to MemForge

Thanks for your interest in contributing to MemForge! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/MemForgeAI/memforge.git
cd memforge

# Install dependencies
npm install

# Start Postgres (pgvector + Apache AGE)
docker compose up postgres -d

# Copy and configure environment
cp .env.example .env

# Start dev server
npm run dev
```

## Making Changes

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feat/your-feature`
3. **Make your changes** and add tests
4. **Run checks:**
   ```bash
   npm run build        # TypeScript compilation
   npm run lint         # ESLint
   npm run format:check # Prettier
   npm test             # Unit tests
   ```
5. **Commit** with a clear message
6. **Open a PR** against `main`

## Code Style

- **TypeScript**: ES modules, strict mode, 2-space indent, single quotes, Prettier defaults
- `camelCase` for variables/functions, `PascalCase` for types/classes
- Always type function parameters and return values
- **SQL**: Parameterized queries only, `snake_case` naming
- **Python** (tests): ruff format, 4-space indent, double quotes, type annotations

## PR Checklist

- [ ] Tests pass (`npm test`)
- [ ] Build succeeds (`npm run build`)
- [ ] Lint passes (`npm run lint`)
- [ ] No secrets or API keys committed
- [ ] Description explains the "why"

## Reporting Issues

Use [GitHub Issues](https://github.com/MemForgeAI/memforge/issues) with the provided templates.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (BSL 1.1).
