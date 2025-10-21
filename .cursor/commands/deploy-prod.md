# Deploy to Production

## Overview
Merge the `staging` branch into `main`. On conflicts, accept changes from `staging`.

## Steps

1. **Fetch latest**
```bash
git fetch origin staging main
```

2. **Switch to main**
```bash
git checkout main
```

3. **Merge staging into main**
```bash
git merge -X theirs staging
```

4. **Push to remote**
```bash
git push origin main
```

5. **Verify**
```bash
git log --oneline -5
```
