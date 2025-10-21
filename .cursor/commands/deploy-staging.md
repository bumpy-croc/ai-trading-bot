# Deploy to Staging

## Overview
Merge the `develop` branch into `staging`. On conflicts, accept changes from `develop`.

## Steps

1. **Fetch latest**
```bash
git fetch origin develop staging
```

2. **Switch to staging**
```bash
git checkout staging
```

3. **Merge develop into staging**
```bash
git merge -X theirs develop
```

4. **Push to remote**
```bash
git push origin staging
```

5. **Verify**
```bash
git log --oneline -5
```
