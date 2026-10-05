#!/usr/bin/env bash
# Rebuild the optional legacy classification datasets on ybforever from public sources.
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
  rm -f "$destination.sha256-ok"
  wget --continue --tries=5 --waitretry=15 --timeout=60 --show-progress -O "$destination" "$url"
  printf '%s  %s\n' "$expected_sha256" "$filename" | (cd "$raw_dir" && sha256sum --check --strict)
  touch "$destination.sha256-ok"
}

extract_tar() {
  local alias="$1"
  local filename="$2"
  local destination="$DATA_ROOT/$alias/extracted"
  mkdir -p "$destination"
  tar -xf "$DATA_ROOT/$alias/raw/$filename" -C "$destination"
}

extract_zip() {
  local alias="$1"
  local filename="$2"
  local destination="$DATA_ROOT/$alias/extracted"
  mkdir -p "$destination"
  python3 - "$DATA_ROOT/$alias/raw/$filename" "$destination" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
}

# These sources are public and their downloaded artifacts are pinned to the
# SHA256 values recorded on the original pro6000 staging host.
fetch cifar100 "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" "cifar-100-python.tar.gz" "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
extract_tar cifar100 "cifar-100-python.tar.gz"

fetch flowers102 "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz" "102flowers.tgz" "2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246"
fetch flowers102 "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat" "imagelabels.mat" "4903e94206bac23bf772aadf06451916df56b58fc483a62db32a97b82656651d"
fetch flowers102 "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat" "setid.mat" "46b8678f91fd95d3c8f4feab80d271a6c834a1dd896fe29fd3e6ad9ce5c8dccd"
extract_tar flowers102 "102flowers.tgz"

fetch pets "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz" "images.tar.gz" "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d"
fetch pets "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz" "annotations.tar.gz" "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91"
extract_tar pets "images.tar.gz"
extract_tar pets "annotations.tar.gz"

fetch dtd "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" "dtd-r1.0.1.tar.gz" "e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205"
extract_tar dtd "dtd-r1.0.1.tar.gz"

fetch eurosat "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1" "EuroSAT_RGB.zip" "b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90"
extract_zip eurosat "EuroSAT_RGB.zip"

fetch resisc45 "https://hf-mirror.com/datasets/taesiri/NWPU-RESISC45/resolve/main/NWPU-RESISC45.zip" "NWPU-RESISC45.zip" "beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53"
extract_zip resisc45 "NWPU-RESISC45.zip"

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

printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$LOG_ROOT/optional_dataset_download_2026-08-17.completed"
