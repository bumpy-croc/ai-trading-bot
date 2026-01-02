# CODE.md

**All instructions in this file must be followed when making code changes.**

This file contains coding guidelines and quality standards

---

## Coding Style & Naming Conventions

### Key Conventions
1. Prioritize **readability, simplicity, and maintainability**.
2. Design for **change**: isolate business logic and minimize framework lock-in.
3. Emphasize clear **boundaries** and **dependency inversion**.
4. Ensure all behavior is **observable, testable, and documented**.
5. **Automate workflows** for testing, building, and deployment.

### General

- Use descriptive variable names
- Use early returns and guard clauses
- Avoid magic numbers. Define constants at module level
- Avoid overly large files
- Avoid obvious within-file duplication
- Avoid deep nesting levels
- Remove debugging and temporary code before commits
- Code should be transparent in its intent
- Keep lines at a readable length
- Avoid unreachable dead code
- Favor composition over inheritance
- Avoid use of goto statements
- Don't use break in inner loops without clear documentation
- Release locks even on exception paths
- Place all user-customizable configuration variables at the beginning of scripts

### Functions
- Keep functions concise
- Don't override function arguments
- Make a function's purpose self-evident
- Don't overuse undocumented anonymous functions
- Functions should always have a doc comment explaining what it does

### Regular Expressions
- Avoid slow regular expressions (nested quantifiers or ambiguous patterns can cause catastrophic backtracking)

### Error Handling
- Handle errors in catch blocks (no empty catch blocks)
- Implement robust error handling
- Prioritize specific exception types over generic ones
- Log errors with sufficient context (e.g., relevant variables, operation attempted)
- Avoid silencing errors unless explicitly requested and justified
- Proactively include input validation and checks for null/undefined/unexpected values

### Classes
- Classes should have single responsibility
- Use one class per file

### Databases
- Avoid SELECT * in SQL queries
- Avoid redundant database indexes

### Math
- Check divisor before division operations

### Comments
- Comment on the goal (why), not the mechanics (what)
- Don't use words like "new", "updated", etc in comments or filenames
- For complex algorithms, include a high-level comment explaining the approach
- Comments must describe the code's current state and purpose, not history of changes
- Use simple present tense to describe what the code *does*
  - Bad: `// New enhanced v2 API.`
  - Good: `// Fetches user data from the v2 API.`

### Types
- Avoid `Any` where possible in type systems
- Type definitions properly, especially for public APIs

### Security

- Never embed actual sensitive information (API keys, passwords, personal data) directly in code
- Always use clear, conventional placeholders (e.g., `YOUR_API_KEY`, `DATABASE_CONNECTION_STRING`)
- Apply input validation and sanitization rigorously, especially on inputs from external sources
- Implement retries, exponential backoff, and timeouts on all external calls

### Documentation

- Maintain documentation in `docs/` to guide team practices and architecture decisions

### Tests
- Keep tests stateless: Use fixtures, avoid global state
- Use the Arrange - Act - Assert (AAA) pattern
- Unit tests should be FIRST (fast, isolated, repeatable, self-validating and timely)

---

## Security & Best Practices

These guidelines are extracted from patterns identified during code reviews. Follow these to avoid common issues.

### Division and Arithmetic

- Always validate divisor is non-zero before division. Use guard clauses or early returns
- Check for NaN and Infinity values in numeric inputs before calculations: `math.isfinite(value)`
- Validate numeric values are positive before using as denominators in financial calculations
- Use epsilon tolerance for float comparisons, never use exact equality: `abs(a - b) < EPSILON`
- Check for negative values when they would produce invalid results (e.g., negative balances, negative prices)

### Array and Index Access

- Always validate array/list indices are within bounds before access: `if 0 <= index < len(array)`
- Check DataFrame is not empty before calling `.iloc[]`, `.min()`, `.max()`
- Validate `split()` results have expected number of elements before indexing
- Use `.get()` for dictionary access when key may not exist

### Timezone Handling

- Always use UTC-aware timezone handling. Never mix or compare naive and UTC-aware dates
- Use `datetime.now(UTC)` consistently, not `datetime.utcnow()` (deprecated)
- Store all timestamps in UTC, convert to local timezone only for display

### Path and File Handling

- Validate user-provided paths with `.resolve()` and parent directory checks to prevent path traversal
- Use Path objects instead of string concatenation for file paths
- Check file/directory exists before operations
- Use atomic writes for file operations to prevent corruption (write to temp, then rename)

### Thread Safety and Concurrency

- Add locks around shared mutable state accessed from multiple threads
- Move callbacks and long-running operations outside lock scope to prevent deadlocks
- Verify threads stopped after join timeout to prevent race conditions
- Release locks in finally blocks to ensure release even on exceptions

### External API Calls

- Add timeout protection to all external HTTP requests and database queries
- Implement circuit breakers for repeated API failures
- Use exponential backoff for retries
- Validate API response types before accessing fields (check if dict, list, etc.)
- Check HTTP response status codes before processing response body

### Resource Management

- Always close sessions, connections, and file handles (use context managers)
- Clean up ONNX sessions and ML model resources to prevent file descriptor leaks
- Use `ExitStack` for managing multiple context managers

### Input Validation

- Validate external inputs at system boundaries (user input, API responses, file contents)
- Check JSON/dict responses are the expected type before key access
- Validate numeric ranges for parameters (e.g., percentages between 0-1)
- Sanitize string inputs used in queries or commands

### Error Handling Patterns

- Never silently swallow exceptions - log with context at minimum
- Elevate critical initialization failures from DEBUG to WARNING/ERROR
- Include relevant variable values in error messages for debugging
- Use specific exception types that callers can catch selectively

### Financial Calculations

- Use consistent fee/slippage calculation across all code paths (shared modules)
- Track entry balance at position creation for accurate P&L calculations
- Validate price values are positive and finite before order calculations
- Protect against negative balance corruption with validation at every update