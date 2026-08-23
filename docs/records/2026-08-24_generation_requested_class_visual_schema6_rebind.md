# Requested-class visual audit schema-6 source rebind

The terminal 100K requested-class visual audit failed closed after both matched
10K sample sets had completed. The failure was not a sample or model failure:
the audit required `sampling.num_classes`, while the native schema-6 sampling
report intentionally records the balanced-modulo class schedule without that
redundant field.

The repair keeps the audit CPU-only and permanently non-authorizing. It now
derives the class count from the exact completed quality-bridge result and
requires agreement across:

- the matched training recipe for CoFiTok and dense identity;
- the terminal classifier and both class-fidelity metric reports;
- both exact sampling-report identities, sampling protocols, prefix budgets,
  sample counts, and sample-set SHA256 values;
- the class-fidelity sampling contract; and
- the independently calibrated real-validation class order.

If a schema-6 sampling report does carry an optional `num_classes`, that value
must agree with the result-bound class count. The original failed status is
historical evidence and must not be overwritten; deployment uses versioned
status/output paths and remains incapable of authorizing training, 300K,
promotion, export, release, or process signals.
