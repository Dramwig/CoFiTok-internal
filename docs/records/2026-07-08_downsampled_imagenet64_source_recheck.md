# Downsampled ImageNet-64 Source Recheck

Date: 2026-07-08

Purpose: decide whether the strict `exact_downsampled_imagenet64_source` gap can
be closed after the first TFDS failure and ImageNet-1K 64x64 HF fallback.

## Sources Checked

- TensorFlow Datasets catalog:
  https://www.tensorflow.org/datasets/catalog/downsampled_imagenet
- TFDS GitHub issue discussing the broken source:
  https://github.com/tensorflow/datasets/issues/4662
- ImageNet old small-image entry:
  http://image-net.org/small/download.php
- ImageNet current download page:
  https://image-net.org/download-images.php
- Academic Torrents candidate:
  https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b

## Remote HTTP Recheck

Host:

```text
pro6000
```

Observed on 2026-07-08:

| URL | observed status |
|---|---|
| `http://image-net.org/small/download.php` | 301 to `https://image-net.org/small/download.php`, then 404 |
| `https://image-net.org/small/train_64x64.tar` | 404 |
| `https://image-net.org/small/valid_64x64.tar` | 404 |
| `https://image-net.org/download-images.php` | 200 HTML download page, not a direct small-image archive |
| `https://www.tensorflow.org/datasets/catalog/downsampled_imagenet` | remote direct connection timed out; with `/etc/network_turbo`, SSL connection timed out |
| `https://academictorrents.com/download/96816a530ee002254d29bf7a61c0c158d3dedc3b` | remote HTTPS connection timed out |
| `https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b` | remote HTTPS connection timed out |

Local Windows HTTP recheck for the Academic Torrents download endpoint also
timed out or closed the TLS connection before headers were returned.

## 2026-07-09 Lightweight Recheck

No data was downloaded; only HEAD requests with short timeouts were used.

| URL | local observed status | `pro6000` observed status |
|---|---|---|
| `https://image-net.org/small/train_64x64.tar` | 404 | 404 |
| `https://image-net.org/small/valid_64x64.tar` | 404 | not retried after train tar remained 404 |
| `https://academictorrents.com/download/96816a530ee002254d29bf7a61c0c158d3dedc3b` | timed out after 20s | timed out connecting to port 443 |

This confirms that the exact canonical route is still unavailable from the
current local/remote network path.

## 2026-07-09 Local And Remote Asset Recheck

No new data was downloaded.

Local raw hub:

```text
D:\datasets_raw_hub\registry\downsampled_imagenet_64\raw
```

was checked from the current Windows workspace. The machine has no accessible
`D:` drive in this session, so no local raw archive can be used to close the
gap.

Remote project assets were checked under:

```text
/root/autodl-tmp/CoFiTok/datasets
```

Relevant observations:

| path | observed state |
|---|---|
| `datasets/downsampled_imagenet_64` | 8 KiB total; contains TFDS metadata and failed `.tmp` download directories only |
| `datasets/downsampled_imagenet_64/tfds/downsampled_imagenet/64x64` | directory exists but contains no materialized split shards |
| `datasets/downsampled_imagenet_64_tfds_inspect.json` | TFDS builder metadata only; `splits: {}` and no examples materialized |
| `datasets/downsampled_imagenet_64_academic_503.html` | cached Squid error page from the Academic Torrents attempt |
| `datasets/imagenet_1k_64x64_hf` | 16 GiB ImageNet-family fallback already materialized; not the exact TFDS source |

This rules out both expected offline recovery paths: there is no local raw hub
copy available in the current environment, and the remote
`downsampled_imagenet_64` directory is only a failed TFDS attempt rather than a
usable dataset.

## 2026-07-09 Completion Recheck

After the earlier HTTP and local-hub attempts failed, the strict source was
completed on `pro6000` through the Academic Torrents magnet route.

Evidence:

| item | value |
|---|---|
| source | `https://academictorrents.com/details/96816a530ee002254d29bf7a61c0c158d3dedc3b` |
| info hash | `96816a530ee002254d29bf7a61c0c158d3dedc3b` |
| download route | `aria2c` magnet + DHT / peer transfer on `pro6000` |
| raw archives | `raw/small/train_64x64.tar`, `raw/small/valid_64x64.tar` |
| extracted splits | `extracted/train_64x64`, `extracted/valid_64x64` |
| split counts | train `1,281,149` PNGs; valid `49,999` PNGs |
| manifest | `docs/experiment_conditions/downsampled_imagenet_64_manifest_summary_2026-07-09.json` |
| detailed record | `docs/experiment_conditions/downsampled_imagenet_64_2026-07-09.md` |

The ImageNet official small-image URLs and the TFDS web route remain broken from
this environment, but the canonical Academic Torrents payload is now present,
checksummed, extracted, and recorded.

## Decision

Treat the strict `downsampled_imagenet_64` source as available as of
2026-07-09 via the Academic Torrents payload. This closes the strict source
availability gap.

Keep the existing `imagenet_1k_64x64_hf` experiment evidence explicitly named
as an ImageNet-family 64x64 fallback. Existing result tables should not be
renamed to `downsampled_imagenet_64` unless those experiments are rerun on the
strict source.

## Next Action

Run exact-source experiments only if a paper revision or reviewer framing
requires replacing the current HF fallback evidence with the strict
`downsampled_imagenet_64` dataset. Otherwise keep the current paper tables
truthfully labeled and proceed with claim hardening.
