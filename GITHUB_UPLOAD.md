# Uploading this repository to GitHub via the terminal

Two parts: a one-time setup, then the actual upload. Commands are for macOS/Linux
(and Git Bash on Windows). Replace `YOUR-USERNAME` and `smallworld-qtn` as needed.

---

## 0. One-time setup (only if you have never used git on this machine)

```bash
# Check git is installed (install from https://git-scm.com if not)
git --version

# Tell git who you are (use your GitHub email)
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

---

## 1. Create the empty repository on GitHub

**Option A — GitHub website (simplest):**
1. Go to https://github.com/new
2. Repository name: `smallworld-qtn`
3. Choose Public or Private. **Do NOT** tick "Add a README", ".gitignore", or
   "license" (this folder already has them).
4. Click **Create repository**. Copy the URL it shows, e.g.
   `https://github.com/YOUR-USERNAME/smallworld-qtn.git`

**Option B — GitHub CLI (`gh`), if installed:**
```bash
gh auth login                 # follow the prompts once
gh repo create smallworld-qtn --public --source=. --remote=origin --push
# If you use Option B, it does steps 2-4 below for you; skip to "Done".
```

---

## 2. Initialize git in this folder and make the first commit

```bash
cd path/to/smallworld-qtn      # the folder that contains README.md and src/

git init
git add .
git status                     # optional: review what will be committed
git commit -m "Initial release: small-world topology from time-series representations"
```

> The included `.gitignore` already excludes data, downloaded files, figures, and
> Python caches, so large/raw files will not be committed.

---

## 3. Connect to GitHub and push

```bash
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/smallworld-qtn.git
git push -u origin main
```

When prompted for a password, GitHub no longer accepts your account password —
use a **Personal Access Token (PAT)** instead:
1. https://github.com/settings/tokens → "Generate new token (classic)"
2. Tick the `repo` scope, generate, and copy the token.
3. Paste it as the password when `git push` asks.

(Or use SSH instead — see "SSH alternative" below — or the `gh` CLI which handles
auth for you.)

---

## Done

Reload your repo page on GitHub; all files should be there and the CI badge will
appear after the first Actions run.

---

## Making changes later

```bash
git add .
git commit -m "Describe what you changed"
git push
```

---

## SSH alternative (no token needed each push)

```bash
# Generate a key once (press Enter for defaults):
ssh-keygen -t ed25519 -C "you@example.com"
cat ~/.ssh/id_ed25519.pub        # copy this and add at:
                                 # https://github.com/settings/keys

# Then use the SSH remote instead of https:
git remote add origin git@github.com:YOUR-USERNAME/smallworld-qtn.git
git push -u origin main
```

---

## Troubleshooting

- **"remote origin already exists"** → `git remote set-url origin <URL>` to fix the URL.
- **"failed to push ... fetch first"** (repo not empty) →
  `git pull origin main --allow-unrelated-histories` then `git push`.
- **Pushed a big file by accident** → add it to `.gitignore`, then
  `git rm --cached path/to/file && git commit -m "remove large file" && git push`.
- **Wrong default branch name** → `git branch -M main` renames it to `main`.
