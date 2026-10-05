#!/usr/bin/env bash
# Continue optional-dataset staging after unavailable legacy mirrors are skipped.
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

fetch svhn "http://ufldl.stanford.edu/housenumbers/train_32x32.mat" "train_32x32.mat" "435e94d69a87fde4fd4d7f3dd208dfc32cb6ae8af2240d066de1df7508d083b8"
fetch svhn "http://ufldl.stanford.edu/housenumbers/test_32x32.mat" "test_32x32.mat" "cdce80dfb2a2c4c6160906d0bd7c68ec5a99d7ca4831afa54f09182025b6a75b"
fetch svhn "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat" "extra_32x32.mat" "a133a4beb38a00fcdda90c9489e0c04f900b660ce8a316a5e854838379a71eb3"
mkdir -p "$DATA_ROOT/svhn/extracted"
cp -f "$DATA_ROOT/svhn/raw/"*.mat "$DATA_ROOT/svhn/extracted/"

fetch stanford_dogs "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar" "images.tar" "a443fc2f9a851aa1a7b8e4fa4af94b8459fce2c9f5a2f4dd4669cf9673a4ec58"
fetch stanford_dogs "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar" "annotation.tar" "662ed36f707be6b195c2fda392d8e153e83af28cedc8e2134a649f198def30fd"
fetch stanford_dogs "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar" "lists.tar" "34b47cacd9a98b5d150e084f24d29391c084c55272295ec65c85651bc35f4d6c"
extract_tar stanford_dogs "images.tar"
extract_tar stanford_dogs "annotation.tar"
extract_tar stanford_dogs "lists.tar"

fetch fgvc_aircraft "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz" "fgvc-aircraft-2013b.tar.gz" "e4e323d410e29f0370c81eabdcbb0e2b813acea1de22891b70b58ff41bfc9834"
extract_tar fgvc_aircraft "fgvc-aircraft-2013b.tar.gz"

printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$LOG_ROOT/optional_dataset_public_recovery_2026-08-17.completed"
