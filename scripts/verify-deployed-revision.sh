#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "verify-deployed-revision: $*" >&2
  exit 1
}

[[ $# == 2 ]] || die "expected PROJECT_ROOT and REVISION"
project_root="$1"
expected_revision="$2"

[[ "$project_root" == /* && -d "$project_root" ]] || die "project root must be an absolute directory"
[[ "$expected_revision" =~ ^[0-9a-f]{40}$ ]] || die "revision must be one lowercase Git commit"
actual_revision="$(git -C "$project_root" rev-parse --verify HEAD 2>/dev/null)" ||
  die "project root is not a readable Git worktree"
[[ "$actual_revision" == "$expected_revision" ]] || die "checked-out revision does not match the deployed revision"
[[ -z "$(git -C "$project_root" status --porcelain --untracked-files=normal)" ]] ||
  die "deployed Git worktree is not clean"
