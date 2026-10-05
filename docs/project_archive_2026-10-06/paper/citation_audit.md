# CoFiTok Citation and Claim Audit

Source audit date: 2026-07-08. Current-paper linkage checked: 2026-07-12.

## Scope and Provenance

All nine keys in `references.bib` are cited by the current venue-neutral and
AAAI-27 main-paper sources. Their arXiv metadata, BibTeX, and DOI resolution
were checked programmatically. The corresponding full arXiv PDFs were also
read at claim level after text extraction with `pdftotext -layout -enc UTF-8`.

Exact PDF URLs, byte counts, and SHA256 values are retained in
`full_pdf_claim_audit_sources.json`; downloaded PDFs are not stored in this
project.

| role | citation keys | audit result |
|---|---|---|
| Pixel diffusion | `ho2020denoising` | Supports Gaussian noising, reverse denoising, and epsilon prediction. |
| Latent diffusion | `rombach2022latentdiffusion` | Supports diffusion in a pretrained autoencoder latent space followed by decoding. |
| Latent/pixel trajectory | `baade2026latentforcing` | Supports jointly processing latent and pixel variables with separately tuned schedules. |
| Ordered tokenizers | `esteves2025spectral`, `bachmann2025flextok` | Supports ordered image tokens, useful prefixes, and partial reconstruction. |
| Nested representations | `kusupati2024matryoshka`, `cai2024matryoshkamultimodal` | Supports nested embedding or visual-token granularity. |
| Closest diffusion-AR work | `wang2025selftok`, `gao2025dar` | Supports the diffusion-to-token connection; D-AR explicitly maps token positions to pixel-space denoising steps. |

## Supported Framing

- CoFiTok factorizes dense pixel-space noise prediction into continuous,
  negative-noise components. The locked MVP evidence supports ordered restricted
  factorization and prefix control; it does not by itself establish a rate-
  compression claim for the full-resolution token fields.
- Each component is expanded by a restricted, condition-free synthesis
  operator; this restriction and the zero/random/shuffle diagnostics support
  the distinction from a powerful decoder.
- Spectral Image Tokenizer, FlexTok, MRL, and M3 concern image or representation
  tokenization, not dense diffusion noise-prediction factorization.
- Selftok and D-AR are the nearest conceptual neighbors and must be acknowledged
  directly. Novelty is the represented object and restricted synthesis
  contract, not the general idea of coarse-to-fine visual tokens.
- Current evidence supports prefix-controllable denoising. It does not support
  a broad unconditional generation-quality or visual-tokenizer SOTA claim.

## Paper-Facing Guardrails

- Do not claim that CoFiTok is the first coarse-to-fine visual-token method.
- Do not describe endpoint-only factorization as the direct dense baseline.
- Do not reintroduce velocity or flow-matching claims without an appropriate
  citation.
- Keep `imagenet_1k_64x64_hf` distinct from strict-source
  `downsampled_imagenet_64`.

## Maintenance

Re-run the claim audit when cited works, related-work wording, or supported
claims change materially. Before submission, verify whether any arXiv `@misc`
entries should be replaced by proceedings metadata; never replace metadata
from memory.
