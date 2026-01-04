# Update Documentation

Update all automated documentation files to reflect recent changes.

## Instructions

1. **Review Recent Changes**
   - Run `git log --oneline -10` to see recent commits
   - Run `git diff --stat HEAD~5` to see files changed recently

2. **Update Changelog** (`docs/changelog.md`)
   - Add entries under `[Unreleased]` section for new features, changes, and fixes
   - Use categories: Added, Changed, Fixed, Removed, Documentation, Infrastructure
   - Move entries to dated sections when releasing
   - Format: `- Description of change (#PR_NUMBER if applicable)`

3. **Update Project Status** (`docs/project_status.md`)
   - Update "Current Focus" with active work
   - Move completed items from "In Progress" to "Completed"
   - Add new planned items discovered during development
   - Update "Last Session Summary" with:
     - Today's date
     - Work completed this session
     - Where you ended (current state)
     - Next steps for future sessions

4. **Update Architecture** (`docs/architecture.md`) - Only if structural changes were made
   - Update component descriptions if new modules added
   - Update directory structure if files moved/created
   - Add to "Recent Architectural Changes" section

5. **Commit Documentation Updates**
   ```bash
   git add docs/changelog.md docs/project_status.md docs/architecture.md
   git commit -m "docs: update automated documentation"
   ```

## When to Run

- After completing a feature or bug fix
- At the end of a development session
- Before creating a pull request
- After merging significant changes

## Output

Report a summary of what was updated in each file.
