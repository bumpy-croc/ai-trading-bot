# CODE.md

**All instructions in this file must be followed when making code changes.**

This file contains coding guidelines and quality standards for this project.

---

## Software Architecture Principles

These foundational principles guide all architectural decisions in this codebase.

### 1. Separation of Concerns

The foundation everything else builds on. Keep distinct responsibilities isolated (presentation, business logic, data access). This makes systems understandable, testable, and changeable. Get this wrong and complexity compounds everywhere.

### 2. Design for Change (Loose Coupling)

Assume requirements will evolve. Depend on abstractions rather than concretions, use interfaces at boundaries, and minimise the ripple effect of changes. This is what keeps a codebase viable at year three versus year one.

### 3. Keep It Simple (YAGNI/KISS)

Resist speculative generality. The cleverest abstraction you don't need yet is technical debt with interest. Build for today's requirements with room to extend, not pre-built extension points.

### 4. Single Responsibility at Every Level

Applies to functions, classes, modules, and services. When something has one clear reason to change, it's easier to understand, test, and replace.

### 5. Define Clear Boundaries and Contracts

Whether microservices or modules, explicit APIs at boundaries let teams work independently and systems evolve in pieces. This includes versioning strategies and backwards compatibility thinking.

### 6. Favour Composition Over Inheritance

Inheritance hierarchies become brittle; composition gives flexibility. This principle extends beyond OOP into how you combine services and modules.

### 7. Design for Failure

Assume networks fail, dependencies go down, and load spikes happen. Circuit breakers, retries with backoff, graceful degradation, and timeouts should be architectural defaults, not afterthoughts.

### 8. Make It Observable

Structured logging, metrics, and tracing aren't optional extras. If you can't see what's happening in production, you can't debug, optimise, or confidently deploy.

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
- Avoid overly large files.
- Avoid obvious within-file duplication
- Avoid deep nesting levels
- Remove debugging and temporary code before commits
- Code should be transparent in its intent.
- Keep lines at a readable length
- Avoid unreachable dead code
- Avoid using magic numbers. All contants should be declared with a descriptive name before its use.
- Favor composition over inheritance
- Avoid use of goto statements
- Don't use break in inner loops (break statements in deeply nested loops make control flow hard to follow without clear documentation.)
- Release locks even on exception paths (every lock acquisition must have a guaranteed release, even when exceptions occur)
- Place all user-customizable configuration variables at the beginning of scripts.

### Functions
- Keep functions concise
- Don't override function arguments
- Make a function's purpose self-evident
- Don't overuse undocumented anonymous functions
- Functions should always have a doc comment explaining what it does


### Regular expressions
- Avoid slow regular expressions (nested quantifiers or ambiguous patterns can cause catastrophic backtracking and performance issues.)

### Error Handling
- Handle errors in catch blocks (no empty catch blocks)
- Implement robust error handling.
- Prioritize specific exception types over generic ones. 
- Log errors with sufficient context (e.g., relevant variables, operation attempted). 
- Avoid silencing errors unless explicitly requested and justified. 
- Proactively include input validation and checks for null/undefined/unexpected values.

### Classes
- Classes should have single responsibility
- Use one class per file

### Databases
- Avoid SELECT * in SQL queries
- Avoid redundant database indexes

### Math
- Check divisor before division operations (division by zero causes runtime crashes and must be prevented with explicit checks)

### Comments
- Comment on the goal (why), not the mechanics (what)
- Don't ever use words like "new", "updated", etc in comments or filenames. 
- For complex algorithms or non-obvious logic, include a high-level comment explaining the approach before the code block
- Comments must describe the code's current state and purpose, not the history of changes made to it. All comments should be written in the simple present tense to describe what the code *does*, not what it *used to do* or *now does*. Examples:
	- **Bad:** `// New enhanced v2 API.`
	- **Good:** `// * Fetches user data from the v2 API.`
	- **Bad:** `// TODO: This was a temporary fix, will rewrite later.`
	- **Good:** `// TODO: Refactor this logic to be more efficient.`

### Types
- Avoid `Any` where possible in type systems
- **Type definitions properly**, especially when dealing with public APIs.

### Security

- Never embed actual sensitive information (API keys, passwords, personal data, specific user-dependent URLs/paths) directly in code.
- Always use clear, conventional placeholders (e.g., `YOUR_API_KEY`, `DATABASE_CONNECTION_STRING`, `PATH_TO_YOUR_FILE`).
- Apply **input validation and sanitization** rigorously, especially on inputs from external sources.
- Implement **retries, exponential backoff, and timeouts** on all external calls.

### Documentation 

- Maintain documentation in `docs/` to guide team practices and architecture decisions.

### Tests
- **Keep tests stateless**: Use fixtures, avoid global state.
- When writing tests, use the Arrange - Act - Assert (AAA) pattern
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
