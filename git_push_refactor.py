#!/usr/bin/env python3
import subprocess, json, os

def run_cmd(cmd, capture=True, check=False):
    """Run command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, check=check)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else str(e)
    except Exception as e:
        return False, "", str(e)

def main():
    # 1) Safety checks - get current branch
    success, branch, _ = run_cmd("git rev-parse --abbrev-ref HEAD")
    if not success:
        branch = "unknown"

    # Check if feature/presentation-guide exists and switch if needed
    target_branch = "feature/presentation-guide"
    if branch != target_branch:
        # Try to checkout target branch
        success, _, _ = run_cmd(f"git checkout {target_branch}")
        if success:
            branch = target_branch

    # 2) Stage changes
    # First check status
    run_cmd("git status -s")

    # Add all relevant files
    files_to_add = [
        "scripts/cli",
        "run_he.py",
        "run_otsu.py",
        "docs/REPO_STRUCTURE.md",
        "README.md",
        "dist/README_submission.md",
        "scripts/make_*"
    ]

    for file_pattern in files_to_add:
        run_cmd(f"git add {file_pattern}")

    # Also add any new files
    run_cmd("git add -A")

    # 3) Commit
    commit_msg = "chore(structure): move CLIs to scripts/cli; update READMEs; add REPO_STRUCTURE; compat stubs"
    commit_success, commit_out, commit_err = run_cmd(f'git commit -m "{commit_msg}"')

    commit_hash = "none"
    if commit_success:
        # Get short hash
        success, hash_out, _ = run_cmd("git rev-parse --short HEAD")
        if success:
            commit_hash = hash_out

    # 4) Rebase-safe push
    pushed = False
    status = "ok"
    error = ""

    if not commit_success and "nothing to commit" in commit_err.lower():
        status = "nothing-to-commit"
    elif not commit_success:
        status = "error"
        error = commit_err.split('\n')[0] if commit_err else "commit failed"

    # Always try to push (even if nothing to commit, to sync)
    if status != "error":
        # Stash any uncommitted changes
        run_cmd("git stash push -u -m 'tmp-refactor'")

        # Try to pull with rebase
        pull_success, _, pull_err = run_cmd(f"git pull --rebase origin {branch}")

        # Pop stash
        run_cmd("git stash pop")

        # Push
        push_success, push_out, push_err = run_cmd(f"git push -u origin {branch}")

        if push_success:
            pushed = True
        else:
            if status == "ok":  # Only override status if not already set
                status = "error"
                error = push_err.split('\n')[0] if push_err else "push failed"

    result = {
        "task": "git_push",
        "branch": branch,
        "commit": commit_hash,
        "pushed": pushed,
        "status": status,
        "error": error
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()