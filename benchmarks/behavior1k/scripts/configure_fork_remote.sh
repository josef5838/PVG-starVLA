#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: configure_fork_remote.sh --owner GITHUB_OWNER [--repo REPOSITORY] [--execute]

Create the fork first at https://github.com/starVLA/starVLA/fork. This script
then adds or verifies a local remote named "fork" without changing "origin".
EOF
}

owner=""
repo=starVLA
execute=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner) owner="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --execute) execute=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$owner" ]] || { usage; exit 2; }

fork_url="git@github.com:$owner/$repo.git"
if git remote get-url fork >/dev/null 2>&1; then
  current="$(git remote get-url fork)"
  if [[ "$current" != "$fork_url" ]]; then
    echo "Existing fork remote is $current, expected $fork_url" >&2
    exit 2
  fi
  echo "Fork remote is already configured: $current"
  exit 0
fi

echo "Will add: git remote add fork $fork_url"
echo "Upstream origin remains: $(git remote get-url origin)"
if [[ "$execute" != true ]]; then
  echo "Dry run only. Re-run with --execute after creating the GitHub fork."
  exit 0
fi
git remote add fork "$fork_url"
echo "Added fork remote: $(git remote get-url fork)"
