# Terminal completion runtime-strict rebind

Date: 2026-08-23

The corrected terminal-system guard and strong-baseline comparison are immutable,
versioned reports. The original terminal completion waiter and its physical replay
builder accepted only the v1 report directory names, so pointing only the waiter at
the corrected sources would still fail inside the builder.

This change adds explicit expected directory-name arguments for the terminal guard,
comparison, and completion output. All default to the original v1 names. A
non-default value must remain a single lower-case report-directory component with
the corresponding role prefix, and every source/output path must still match its
declared directory exactly.

The audit remains CPU-only and permanently non-authorizing. It cannot launch
training or sampling, load GPU work, authorize 300K scaling, promote, export,
release, or signal any process. Existing v1 and failed evidence remains immutable.
