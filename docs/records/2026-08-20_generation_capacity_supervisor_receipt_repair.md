# Capacity supervisor receipt repair (2026-08-20)

## Outcome

The failed six-supervisor migration was repaired without modifying the formal
checkout, signaling an existing process, or launching GPU work. The four
restored upstream supervisors remained alive. Missing post-evaluation and
finalization supervisors were relaunched from the hardened control checkout
and reached their expected waiting states.

This is a control-plane continuity result. It does not change the scientific
protocol, model weights, frozen paper evidence, or generation-quality claim.

## Source-bound repair

- Repair revision: `a963ff4bb91fd837f67856be996810ff45958712`
- Repair tree: `248685d888e74c7844ba0aedd3d858e9c67e3162`
- Plan SHA256: `fa7343a704546dc6eacecdc5dc0b17e9dc0a8b2e93e6bddbc00bad4315adb190`
- Approval SHA256: `fd6af61cccac751c66eef30965b1ccb00ea7a3c6083c2eab623b8eb0b46bc720`
- Execution SHA256: `50aeed70af7f32b29fdd4f9802fd21181516060e51431bbdc63bcaa974e30545`
- Execution status: `pass`

The plan bound the unchanged formal checkout at revision `1ebcc152`, tree
`659fa947`, tracked clean, with full porcelain count `89` and SHA256
`18e5981f22a2ac255c7f60343429daa6ceea86412b1d7bd09bc474f2faf74004`.

## Receipt lineage

| Receipt | Byte-preserving archive SHA256 | New canonical SHA256 |
|---|---|---|
| training | `2ffb1518c14034f4879c5500f933768e7406c44b52bf0d66ac75b3c3b31696ce` | `4a86f1c399cf4dcc3457d9a63db9f3cca0d4dfe028e22c551693ac2bf3f54267` |
| posteval | `1dd2326555fce015be4c28063edc8f34d45a66c8e311555f929273d9755b6e1c` | `d97552388ef71ffb4c91b623eae7df6209e5100748193e1631fceae7ad375f3e` |
| finalization | `5948f31da3ea6b4614e893ccf2cab79297e56a9d39eaed9deaf246ebc89635a5` | `9e4cc480efabaf033e1c291779a0f10a899d590e7fba44286fd9722bef8315fe` |

The posteval argv is bound to the new training receipt SHA. The finalization
argv is bound to both the new training and posteval receipt SHAs. Missing
legacy incremental-bundle sources are preserved explicitly in supersession
metadata rather than silently treated as present.

## Runtime verification

- Existing training supervisor: PID `351260`, waiting.
- Repaired posteval supervisor: PID `409865`, waiting.
- Repaired finalization supervisor: PID `409956`, waiting.
- Both new processes have `PPID=1`, `PGID=SID=PID`, the hardened checkout as
  cwd, and inherited lifetime-lock fd `4`.
- Independent nonblocking lock probes were rejected for both locks.
- Linux targeted migration/repair tests: `13 passed`.
- The v2 capacity lineage observer recognized both new PIDs and remained
  `running` on its next heartbeat.
- Active matched dense PID `79894` advanced through step `46,250` during the
  repair and remained the only GPU process.

## Evidence

The synchronized evidence directory is:

`artifacts/reports/generation/capacity_pipeline_supervisor_receipt_repair_2026-08-20/`

The machine-readable summary is `verification_manifest.json`; exact plan,
approval, execution, archived receipts, rotated canonical receipts, and live
status snapshots are retained beside it.
