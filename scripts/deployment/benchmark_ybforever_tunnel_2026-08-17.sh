#!/usr/bin/env bash
set -euo pipefail

bytes_mb="${1:-8}"
streams="${2:-1}"
TIMEFORMAT='tunnel_elapsed_seconds=%3R'
time (
  for _ in $(seq 1 "$streams"); do
    dd if=/dev/zero bs=1M count="$bytes_mb" status=none | \
      ssh -i /root/.ssh/cofitok_ybforever_push_20260817 \
        -o BatchMode=yes -o StrictHostKeyChecking=yes -p 22179 \
        yubohuang@127.0.0.1 'dd of=/dev/null bs=1M status=none' &
  done
  wait
)
