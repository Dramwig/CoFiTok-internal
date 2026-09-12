# Frozen gradient assay: prepared, not authorized

`preparation.json` is a byte-preserving copy of the immutable pro6000 artifact:

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/.frozen_loss_gradient_attribution_v1.control/preparation.json
bytes: 51660
sha256: 8cc706a237812d137389b44985c599789bbca38d3cfa02e8ceb8c07c26cd422b
mode: 0444
mtime_ns: 1789233150586218883
```

Exact implementation revision: `ad237efa810df8677b4187110a723279d47f9d4b`.
Tree: `e5b0def15e93df13b845abbc905cff7eda7bc486`.
The independent validator physically replayed the preparation and preserved its
mtime. Stage approval, execution authorization and result directory were absent.
No real checkpoint was deserialized and no real gradient/GPU measurement ran.

This file is not an approval or result. See the accompanying September 13
guarded-executable record and the protocol for the selected-example and
independent-scalar-validation limitations.
