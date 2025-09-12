# Code Quality standards

## Remove debugging and temporary code before commits

### Languages it applies to: all

Code that bypasses logic, outputs debug info, or stops execution for debugging was likely left behind accidentally during development.

## Detect potentially malicious code patterns

### Languages it applies to: all

Code should be transparent in its intent. Deliberate obfuscation or hiding techniques suggest malicious intent or backdoors.

## Release locks even on exception paths

### Languages it applies to: Python, Javascript, Typescript, Java, C/C++, Go

Every lock acquisition must have a guaranteed release, even when exceptions occur.

## Don't place assignments inside conditionals

### Languages it applies to: Python, Javascript, Typescript, PHP

Mixing assignment and condition logic makes code error-prone and harder to understand. Separate assignments from logical checks.

## Handle errors in catch blocks

### Languages it applies to: all

Empty catch blocks silently swallow errors, making debugging difficult.

## Guard against slow regular expressions

### Languages it applies to: all

Regular expressions with nested quantifiers or ambiguous patterns can cause catastrophic backtracking and performance issues.

## Avoid deep nesting levels

### Languages it applies to: All

Deep nesting makes code hard to read and understand.

## Classes should have single responsibility

### Languages it applies to: all

Classes handling multiple concerns violate the Single Responsibility Principle.

## Eiminate obvious within-file duplication

### Languages it applies to: all

Duplicated code blocks increase maintenance burden and the risk of inconsistent updates.

## Avoid unintended global variable caching

### Languages it applies to: Python, Javascript, Typescript

In Node.js and Python servers, global variables persist across requests, causing data leaks and race conditions.

## Avoid SELECT * in SQL queries

### Languages it applies to: SQL

SELECT * in production code makes applications fragile to schema changes and obscures data dependencies.

## Keep functions concise

### Languages it applies to: all

Long functions are difficult to understand, test, and maintain.

## Don't override function arguments

### Languages it applies to: all

Reassigning function parameters can confuse callers and make debugging difficult.

## One class per file

### Languages it applies to: all

Multiple classes in a single file make code organization unclear and harder to navigate.

## Use early returns and guard clauses

### Languages it applies to: all

Deep nesting and late parameter validation make functions harder to read and maintain.

## Avoid overly large files

### Languages it applies to: all

Large files with multiple responsibilities are hard to maintain.

## Avoid redundant database indexes

### Languages it applies to: SQL

Overlapping database indexes waste storage and slow down writes.

## Use descriptive variable names

### Languages it applies to: all

Very short variable names make code unclear.

## Don't use break in inner loops

### Languages it applies to: all

Break statements in deeply nested loops make control flow hard to follow without clear documentation.

## Don't overuse undocumented anonymous functions

### Languages it applies to: all

Large anonymous functions without documentation are difficult to understand and reuse.

## Make a function's purpose self-evident

### Languages it applies to: all

Functions without clear naming, documentation, or context are difficult for other developers to understand and use.

## Check divisor before division operations

### Languages it applies to: all

Division by zero causes runtime crashes and must be prevented with explicit checks.

## Functions without explanatory comments

### Languages it applies to: all

Functions without comments are hard to understand for other developers.

## Comment on the goal (why), not the mechanics (what)

### Languages it applies to: all

Comments that simply restate what the code does provide no additional value and can become outdated.

## Favor composition over inheritance

### Languages it applies to: all

Deep inheritance hierarchies create tight coupling and make code harder to understand and maintain.

## Avoid use of goto

### Languages it applies to: all

The goto statement creates unstructured control flow that makes code difficult to follow and maintain.

## Keep lines at a readable length

### Languages it applies to: all

Lines that are too long are difficult to read and navigate, especially on smaller screens.

## Remove lingering TODO/FIXME comments

### Languages it applies to: all

Unresolved TODO and FIXME comments indicate incomplete work that can accumulate over time. While these are useful during development, production code should avoid lingering placeholders.

## Remove unreachable dead code

### Languages it applies to: all

Unreachable code is confusing, untestable, and should be removed.

## Use named arguments for clarity

### Languages it applies to: Python, PHP

Named arguments make code self-documenting and prevent parameter order mistakes.

## Avoid dynamic variable names

### Languages it applies to: PHP

Dynamic variable names (variable variables) can lead to hard to maintain code and unexpected behaviour. Their usage is usually the result of a typo.

## Wrap array filtering results with array_values()

### Languages it applies to: PHP

Functions like array_filter() preserve original keys, which can cause bugs when code expects sequential numeric indexes starting from 0.

## Type hints for public APIs

### Language it applies to: All

Avoid `Any` where possible in type systems and provide proper type definitions, especially when dealing with public APIs.