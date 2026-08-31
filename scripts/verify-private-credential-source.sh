#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "verify-private-credential-source: $*" >&2
  exit 1
}

[[ $# == 1 ]] || die "expected one credential source path"
credential_source="$1"

for required_command in dirname getfacl getent grep id realpath stat; do
  command -v "$required_command" >/dev/null || die "missing prerequisite: $required_command"
done

[[ "$credential_source" == /* ]] || die "credential source must be an absolute path"
[[ "$credential_source" =~ ^/[A-Za-z0-9._/@+/-]+$ ]] ||
  die "credential source contains unsupported path characters"
[[ "$(realpath -e -- "$credential_source" 2>/dev/null || true)" == "$credential_source" ]] ||
  die "credential source must not traverse symlinks or aliases"
[[ -f "$credential_source" && ! -L "$credential_source" ]] ||
  die "credential source must be one regular non-symlink file"

current_uid="$(id -u)"
primary_gid="$(id -g)"
read -r credential_uid credential_mode credential_links credential_type < <(
  stat -c '%u %a %h %F' -- "$credential_source"
)
[[ "$credential_uid" == "$current_uid" && "$credential_links" == 1 && "$credential_type" == "regular file" ]] ||
  die "credential source must be an owner-private single-link regular file"
(( (8#$credential_mode & 077) == 0 && (8#$credential_mode & 0400) != 0 )) ||
  die "credential source must be readable only by its owner"
if getfacl -cp -- "$credential_source" | grep -Eq '^(default:|user:[^:]|group:[^:]|mask:)'; then
  die "credential source must not carry an extended access ACL"
fi

# A group-writable ancestor is safe only when the group is the caller's private
# primary group. A sticky root/current-user directory such as /tmp is also safe
# because another account cannot replace an entry it does not own.
primary_group_is_private=1
while IFS=: read -r _ _ account_uid account_gid _; do
  if [[ "$account_gid" == "$primary_gid" && "$account_uid" != "$current_uid" ]]; then
    primary_group_is_private=0
  fi
done < <(getent passwd)
primary_group_record="$(getent group "$primary_gid" || true)"
if [[ -z "$primary_group_record" ]]; then
  primary_group_is_private=0
else
  primary_group_members="${primary_group_record##*:}"
  IFS=',' read -r -a group_members <<<"$primary_group_members"
  for member in "${group_members[@]}"; do
    [[ -z "$member" ]] && continue
    member_uid="$(id -u "$member" 2>/dev/null || true)"
    if [[ -z "$member_uid" || "$member_uid" != "$current_uid" ]]; then
      primary_group_is_private=0
    fi
  done
fi

directory="$(dirname -- "$credential_source")"
while :; do
  read -r directory_uid directory_gid directory_mode directory_type < <(
    stat -c '%u %g %a %F' -- "$directory"
  )
  [[ "$directory_type" == "directory" ]] || die "credential directory chain is invalid"
  [[ "$directory_uid" == 0 || "$directory_uid" == "$current_uid" ]] ||
    die "credential directory chain is owned by another account"
  if getfacl -cp -- "$directory" | grep -Eq '^(default:|user:[^:]|group:[^:]|mask:)'; then
    die "credential directory chain must not carry extended access ACLs"
  fi

  directory_bits=$((8#$directory_mode))
  sticky=$((directory_bits & 01000))
  if (( (directory_bits & 0002) != 0 && sticky == 0 )); then
    die "credential directory chain is writable by another account"
  fi
  if (( (directory_bits & 0020) != 0 && sticky == 0 )); then
    [[ "$directory_gid" == "$primary_gid" && "$primary_group_is_private" == 1 ]] ||
      die "credential directory chain is writable by another account"
  fi

  [[ "$directory" == / ]] && break
  directory="$(dirname -- "$directory")"
done
