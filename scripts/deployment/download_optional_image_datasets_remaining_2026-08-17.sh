#!/usr/bin/env bash
# Complete the optional legacy datasets whose sources require independent handling.
set -euo pipefail

readonly DATA_ROOT="${COFITOK_DATA_ROOT:-/home/yubohuang/zixi/CoFiTok/datasets}"
readonly LOG_ROOT="${COFITOK_DEPLOY_LOG_ROOT:-/home/yubohuang/zixi/CoFiTok/deployment_logs}"
mkdir -p "$LOG_ROOT"

fetch() {
  local alias="$1"
  local url="$2"
  local filename="$3"
  local expected_sha256="$4"
  local raw_dir="$DATA_ROOT/$alias/raw"
  mkdir -p "$raw_dir"
  local destination="$raw_dir/$filename"
  if [[ -f "$destination" ]] && [[ "$(sha256sum "$destination" | awk '{print $1}')" == "$expected_sha256" ]]; then
    return
  fi
  wget --continue --tries=5 --waitretry=15 --timeout=60 --show-progress -O "$destination" "$url"
  printf '%s  %s\n' "$expected_sha256" "$filename" | (cd "$raw_dir" && sha256sum --check --strict)
}

extract_tar() {
  mkdir -p "$DATA_ROOT/$1/extracted"
  tar -xf "$DATA_ROOT/$1/raw/$2" -C "$DATA_ROOT/$1/extracted"
}

extract_zip() {
  mkdir -p "$DATA_ROOT/$1/extracted"
  python3 - "$DATA_ROOT/$1/raw/$2" "$DATA_ROOT/$1/extracted" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
}

fetch caltech101 "https://data.caltech.edu/api/records/mzrjq-6wc02/files/caltech-101.zip/content" "caltech-101.zip" "331234750fc7f77520e50d9565e8b6907b03565e30320da1401130db08c61f91"
extract_zip caltech101 "caltech-101.zip"
tar -xzf "$DATA_ROOT/caltech101/extracted/caltech-101/101_ObjectCategories.tar.gz" -C "$DATA_ROOT/caltech101/extracted"

fetch cub_200_2011 "https://data.caltech.edu/api/records/65de6-vp158/files/CUB_200_2011.tgz/content" "CUB_200_2011.tgz" "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081"
extract_tar cub_200_2011 "CUB_200_2011.tgz"

# The historical project staging contains the official test-only GTSRB scope.
# It is intentionally copied only if public retrieval has failed; see the
# deployment record for the exact source provenance and checksum verification.
printf 'caltech101 and cub_200_2011 completed %s\n' "$(date --iso-8601=seconds)" > "$LOG_ROOT/optional_dataset_download_remaining_2026-08-17.completed"
