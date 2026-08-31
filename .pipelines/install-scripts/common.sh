# common.sh — shared helpers for reading pinned versions from a cgmanifest.json and shallow-fetching a pinned
# commit. Sourced by the other install-*.sh scripts in this directory (and exachem/install-exachem.sh); not
# meant to be run directly.

# Resolve a pinned git dependency's commitHash from cgmanifest.json by matching a substring of the registered
# repositoryUrl.
get_commit_hash() {
    local manifest="$1" repo_pattern="$2"
    python3 -c "
import json
with open('$manifest') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'git' and '$repo_pattern' in comp['git'].get('repositoryUrl', ''):
        print(comp['git']['commitHash'].strip())
        break
"
}

# Resolve a pinned git dependency's tag (if any) from cgmanifest.json.
get_tag() {
    local manifest="$1" repo_pattern="$2"
    python3 -c "
import json
with open('$manifest') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'git' and '$repo_pattern' in comp['git'].get('repositoryUrl', ''):
        print(comp['git'].get('tag', ''))
        break
"
}

# Resolve a pinned git dependency's repositoryUrl from cgmanifest.json.
get_repo_url() {
    local manifest="$1" repo_pattern="$2"
    python3 -c "
import json
with open('$manifest') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'git' and '$repo_pattern' in comp['git'].get('repositoryUrl', ''):
        print(comp['git']['repositoryUrl'])
        break
"
}

# Resolve a download URL for an "other" (non-git) type component from cgmanifest.json, by component name.
get_download_url() {
    local manifest="$1" name="$2"
    python3 -c "
import json
with open('$manifest') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'other' and comp['other'].get('name') == '$name':
        print(comp['other']['downloadUrl'])
        break
"
}

# Resolve the declared SHA-1 (`hash`) for an "other" (non-git) type component from cgmanifest.json, by
# component name. Not every "other" entry declares one (e.g. Boost doesn't); callers must check for empty.
get_hash() {
    local manifest="$1" name="$2"
    python3 -c "
import json
with open('$manifest') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'other' and comp['other'].get('name') == '$name':
        print(comp['other'].get('hash', ''))
        break
"
}

# Download an "other" type cgmanifest component and verify its SHA-1 against the declared `hash`, so a
# compromised or silently-changed upstream release asset is caught before it's extracted and built. Fails
# loudly if the component has no downloadUrl or no declared hash. Downloads into the current directory (or
# $3 if given) and prints the downloaded file's path on success, for capture via command substitution.
download_and_verify() {
    local manifest="$1" name="$2" dest="${3:-}"
    local url hash actual

    url="$(get_download_url "$manifest" "$name")"
    if [[ -z "$url" ]]; then
        echo "ERROR: no downloadUrl for '$name' in $manifest" >&2
        return 1
    fi

    hash="$(get_hash "$manifest" "$name")"
    if [[ -z "$hash" ]]; then
        echo "ERROR: no hash declared for '$name' in $manifest -- cannot verify the download" >&2
        return 1
    fi

    [[ -n "$dest" ]] || dest="$(basename "$url")"
    wget -q "$url" -O "$dest"

    actual="$(shasum -a 1 "$dest" | awk '{print $1}')"
    if [[ "$actual" != "$hash" ]]; then
        echo "ERROR: SHA-1 mismatch for '$name' ($url): expected $hash, got $actual" >&2
        rm -f "$dest"
        return 1
    fi

    echo "$dest"
}

# Shallow-fetch a single pinned commit instead of a full clone. `git clone --branch` only resolves branch/tag
# ref names, not an arbitrary commit SHA, so several of our pins (raw commits, not tags) need this instead;
# GitHub allows fetching any reachable SHA this way.
shallow_checkout() {
    local repo_url="$1" commit="$2" dest="$3"
    mkdir -p "$dest"
    git -C "$dest" init -q
    git -C "$dest" remote add origin "$repo_url"
    git -C "$dest" fetch -q --depth 1 origin "$commit"
    git -C "$dest" checkout -q FETCH_HEAD
}
