---
name: update-code-rules
description: Update CODE.md with lessons learned from recent work. Use after long sessions, code reviews, debugging, or when patterns of mistakes emerge. Trigger phrases include "update code rules", "update CODE.md", "lessons learned", "add coding rules", "what did we learn".
---

# Update CODE.md with Lessons Learned

Analyse recent work in this session (reviews, fixes, debugging, new features) and distill reusable coding rules into `CODE.md`.

## Process

1. **Gather context.** Read the current `CODE.md` and recent git history:
   ```
   git log --oneline -30
   git log --format="%s%n%b" -15
   ```
   Also review any review comments, fix iterations, or debugging cycles from the current session.

2. **Identify patterns.** Look for:
   - Bugs that were caught in review or testing — what rule would have prevented them?
   - Mistakes repeated across multiple files or sessions
   - Non-obvious pitfalls specific to this codebase's tech stack
   - Thread safety, state management, or data integrity lessons
   - Patterns that worked well and should be codified

3. **Filter ruthlessly.** Only add rules that are:
   - **Generic** — applicable beyond the specific feature. "Use `or` for null-safe dict access" not "Use `or` for Binance commission fields"
   - **Non-obvious** — an experienced developer might not think of it. Don't add "write tests" or "handle errors"
   - **Actionable** — tells the reader exactly what to do, not just what to avoid
   - **Not already covered** — check existing rules first. Update/strengthen existing rules rather than duplicating

4. **Write rules.** Follow these style conventions:
   - **Imperative mood, present tense.** "Lock shared state" not "Shared state should be locked"
   - **One line per rule.** Use `—` for inline clarification. No multi-paragraph explanations
   - **Lead with the action.** "Snapshot values before mutation" not "When you need to compare state..."
   - **Include the WHY only when non-obvious**, after a `—`. "Cap collections — prevents memory leaks"
   - **Use code snippets sparingly** — only when the pattern is genuinely hard to express in prose
   - **No project-specific language.** "Exchange API" not "Binance API". "Rolling buffer" not "KlineBuffer"

5. **Place rules in the right section.** Match existing CODE.md structure:
   - Thread safety → Thread Safety & Concurrency
   - API handling → External API Calls
   - Data validation → Input Validation
   - Financial correctness → Arithmetic & Financial Calculations
   - Create a new section only if 3+ rules don't fit anywhere

6. **Enforce the 500-line cap.** If CODE.md exceeds 500 lines after additions:
   - Merge overlapping rules
   - Remove rules that are now common knowledge to the team
   - Consolidate verbose rules into terser form
   - Remove code examples that can be expressed as one-line rules

7. **Present changes.** Show the user:
   - Each new/modified rule with a one-line rationale (which bug or review finding inspired it)
   - Any rules removed or consolidated to stay under the cap
   - Ask for approval before writing

## Anti-patterns — do NOT add rules like:

- ❌ "Always write tests" (too obvious)
- ❌ "Handle the PENDING_CANCEL status from Binance" (too specific)
- ❌ "Remember to check for None" (too vague)
- ❌ Rules that duplicate existing entries
- ❌ Multi-sentence explanations — compress to one line
- ❌ Rules only relevant to one file or feature

## Good examples:

- ✅ "Snapshot mutable values before calling a mutation, then compare. The 'before' reference may alias the mutated object."
- ✅ "`dict.get(key, default)` returns `None` when the key exists with JSON `null`. Use `or`: `float(d.get('n') or 0)`."
- ✅ "On reconnect, pass a fresh callback — stale callbacks may reference stopped consumers."
- ✅ "Gate producer→consumer queues with a lock-protected `_closed` flag."
