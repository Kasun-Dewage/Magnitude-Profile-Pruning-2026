# Setup Guide: Publishing This Repo from VS Code

A step-by-step walkthrough for getting this repository from a zip file onto GitHub.

## 1. Unzip and open in VS Code

1. Download `Magnitude-Profile-Pruning-2026.zip` and extract it (right-click → *Extract All* on Windows, or double-click on macOS).
2. Open **VS Code** → `File` → `Open Folder…` → select the extracted `Magnitude-Profile-Pruning-2026` folder.
3. Open the integrated terminal: `Terminal` → `New Terminal` (or `` Ctrl+` ``).

## 2. Initialize git and make the first commit

```bash
git init
git add .
git commit -m "Initial commit: MP/MP-G pruning code, results, and paper"
git branch -M main
```

## 3. Create the GitHub repository

**Option A — on github.com:**
1. Go to https://github.com/new
2. Repository name: `Magnitude-Profile-Pruning-2026`
3. Leave it **empty** (no README/license — this repo already has them). Click *Create repository*.
4. Connect and push:
```bash
git remote add origin https://github.com/Kasun-Dewage/Magnitude-Profile-Pruning-2026.git
git push -u origin main
```

**Option B — with GitHub CLI (`gh`):**
```bash
gh auth login
gh repo create Magnitude-Profile-Pruning-2026 --public --source=. --remote=origin --push
```

## 4. Verify

Visit https://github.com/Kasun-Dewage/Magnitude-Profile-Pruning-2026 — the README renders on the front page, code in `src/`, results in `results/`, paper in `paper/`.

## 5. Everyday workflow

```bash
git status                  # see what changed
git add <file>              # stage changes
git commit -m "message"     # commit
git push                    # upload to GitHub
git pull                    # download latest
```

## Recommended VS Code extensions

- **Python** (ms-python.python)
- **GitLens** — richer git history
- **GitHub Pull Requests and Issues**
- **Markdown Preview** (built in: `Ctrl+Shift+V` to preview README.md)
