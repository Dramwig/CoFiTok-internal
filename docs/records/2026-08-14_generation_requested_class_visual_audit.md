# 2026-08-14 requested-class visual audit

## Outcome

A deterministic real/CoFiTok/dense panel makes the 50K pair's failure mode
directly visible. Both generated rows are dominated by high-frequency colored
speckle, unstable texture, and strong color distortion. Some images preserve a
coarse silhouette or scene layout, but none of the 16 fixed columns provides a
clear, unambiguous match to its requested fish or bird class.

CoFiTok and dense frequently preserve similar coarse geometry for the same
global index. The panel therefore does not isolate the ordered factorization as
the failure source. Together with the near-chance class-fidelity scores and the
successful `77.7%/94.1%` real-validation classifier calibration, the direct
evidence supports a shared undertraining, capacity, or generation-stability
bottleneck. It does not distinguish among those mechanisms.

## Deterministic protocol

- Sampling source: existing matched 10K confirmation, global window
  `10000..19999`.
- Selection: the first 16 contiguous global indices, fixed independently of
  image content and metrics.
- Row order: real validation / CoFiTok K8 / dense identity.
- Real reference: the lexicographically first image in each requested WNID,
  already bound by the passed classifier calibration.
- Generated protocol: EMA, DDIM-100, CFG `1.5`, guidance rescale `1.0`, bf16,
  seed `0`, balanced-modulo classes.
- Requested categories: tench, goldfish, great white shark, tiger shark,
  hammerhead, electric ray, stingray, cock, hen, ostrich, brambling,
  goldfinch, house finch, junco, indigo bunting, and robin.

The visual-audit implementation ran from revision
`2e86ed24925fa958c86dc9bcfa8bc5f029b3a1d8`, tree
`e99296e50806793b103d77fc868df97811eab03e`, branch
`scale/generation-requested-class-visual-audit-v1`. Its focused and regression
suite passed `27/27`, and an identical `--resume` invocation revalidated and
reused the completed report and panels.

## Evidence identities

- Raw visual-audit report: `24,853` bytes, SHA256
  `cccd7a35e5797178ca901217bf7f84cb1064866bb558b9ba67ac5031e6906e95`.
- Panel `10000..10007`: `3,670,755` bytes, SHA256
  `084aa7bb734fa83070a881a13456c803602ad29425d1b0547522769abead8066`.
- Panel `10008..10015`: `3,823,687` bytes, SHA256
  `5a0a73aa3dbde5a867af4174fd81cfd3709565b2553cb1b0a7b96eb83bee89be`.
- CoFiTok sample set SHA256:
  `4fbeb7eb443c1d3996ecf805933aad0ff927cf97b2dbf8ba6025c41fa1225f76`.
- Dense sample set SHA256:
  `29ff181d6c695b8cafca54ec8aeddc43d435ed76ca832f7048e0d47a319014c7`.
- Classifier-calibration report SHA256:
  `a4c68b9b1c4dffda89f622887455f46d554fdb0c306d467fecb51d8f5abe6132`.

## Scope and decision

This is a qualitative, non-authorizing diagnostic. It does not replace formal
FID/IS/precision/recall, class-fidelity qualification, or the frozen promotion
gate, and it authorizes no training, 300K launch, promotion, or release.

The next experiment remains the queued matched full-data base128 100K quality
bridge under the frozen terminal protocol. If its terminal samples retain this
artifact family and near-chance requested-class alignment, the locked follow-up
decision should activate the prepared matched approximately-250M 10K capacity
probe rather than another CFG-only sweep.
