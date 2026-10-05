import os,json,time,hashlib,subprocess,socket
from pathlib import Path
from datetime import datetime,timezone

PY="/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10"
Q=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1")
R=Q/"reports"
STAMP="2026-08-21_server_restart_v1"
REC=R/("downstream_waiter_recovery_"+STAMP)
REC.mkdir(parents=True,exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def git(cwd):
    def g(*a):
        return subprocess.check_output(["git","-C",str(cwd),*a],text=True).strip()
    return {
        "revision":g("rev-parse","HEAD"),
        "tree":g("rev-parse","HEAD^{tree}"),
        "branch":g("branch","--show-current"),
        "dirty":bool(g("status","--porcelain","--untracked-files=no")),
    }

GITS={
"controller":("/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-controller-identity-guard-57b2897/CoFiTok-internal","57b2897a4cfbcde09bb24f3082d7fe3c67b83960","bcc79d4a6c2cba1c8481a0f49956e35739922e29","analysis/generation-quality-bridge-controller-identity-guard-v1"),
"fairness":("/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-runtime-compute-fairness-85bf1d6/CoFiTok-internal","85bf1d6ae5b44f401153d0d1e390fc3973ff4bed","7900c5617c28c18e328c93aba4c547a3889f1aa7","analysis/generation-quality-bridge-runtime-compute-fairness-v1"),
"runtime_claim":("/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-runtime-claim-guard-6fdeebe/CoFiTok-internal","6fdeebe7c0ca6b2f7fbe593b3690f14d36f98e46","bc41a4df45b83cef3c81cc63aad1668c49b85a75","analysis/generation-quality-bridge-runtime-claim-guard-v1"),
"visual":("/root/autodl-tmp/CoFiTok/checkouts/terminal-visual-audit-waiter-c1abf65/CoFiTok-internal","c1abf65fdafb8e198a8f1ac59c83b3038a9702b5","3362a6dc939ae5d907103211db41eaa88851f1d2","scale/generation-terminal-visual-audit-waiter-v1"),
"distribution":("/root/autodl-tmp/CoFiTok/checkouts/terminal-distribution-support-37cb1fe/CoFiTok-internal","37cb1fe474197b8ec8c2f94c6cc97cfd11fc29f0","5a214941de008cf2782d568d749097134d9364bc","analysis/generation-terminal-distribution-support-v1"),
"exposure":("/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-exposure-routing-85e3ece/CoFiTok-internal","85e3ece1196fd318cd6823439824e19fca4275a3","1bf21fa0c44cec90011287b78f6f2ec14ed5eb17","analysis/generation-quality-bridge-exposure-routing-v1"),
"training_exposure":("/root/autodl-tmp/CoFiTok/checkouts/terminal-exposure-waiter-e6f083d","e6f083d01062d594c32b1d595a77f0feb81a0ac4","f5691656c0d66a8b4802e2c5f477c8cca0274143","analysis/generation-terminal-exposure-binding-v1"),
"factor":("/root/autodl-tmp/CoFiTok/checkouts/factorization-quality-regression-8749138/CoFiTok-internal","8749138f4b8e144ccbcae0f897e3237dc5aacada","994c0837c7c3e6f3a2f389cb0a6a351f06d89757","scale/generation-factorization-quality-regression-v1"),
"preceding_uncertainty":("/root/autodl-tmp/CoFiTok/checkouts/matched-uncertainty-waiter-f161fe3/CoFiTok-internal","f161fe31453231b7c126d10cfbe78344f10ff463","187ead45f9418f8a93f28da6dc229ea78c24a425","analysis/generation-matched-uncertainty-v1"),
"evaluator":("/root/autodl-tmp/CoFiTok/checkouts/matched-uncertainty-1c8ef20/CoFiTok-internal","1c8ef207cb6d79850d73a45abc345fc421e6aa7f","a09f14a0eca44af6db8e6781863a463163ddf646",""),
"terminal_uncertainty":("/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-terminal-uncertainty-e9a424b/CoFiTok-internal","e9a424b36bc5b951a9a27d7e1dc7f7afc650f706","4f4cf5015ceb4be426e314bbd224778860e98d00","analysis/generation-quality-bridge-terminal-uncertainty-v1"),
"claim_qualification":("/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-claim-qualification-013beb7/CoFiTok-internal","c42ac96c6628ff71f7c67c8957c86ea0523aea0b","f94f2deb974b0a347b3a647e68dc3dcba22a0d63","analysis/generation-quality-bridge-claim-qualification-v1"),
"claim_language":("/root/autodl-tmp/CoFiTok/checkouts/quality-bridge-claim-language-guard-waiter-5fab079/CoFiTok-internal","5fab079a7b48024b05e2a306b95040baf839e276","3f5ae4b7f79a26bb17183caa04394070c5e9dcd8","analysis/generation-quality-bridge-claim-language-guard-waiter-v1"),
"terminal_system":("/root/autodl-tmp/CoFiTok/checkouts/terminal-system-guard-8ec9a09/CoFiTok-internal","8ec9a09ddcd981c1ffbd06b6bda81386b33321de","344f6365d2e962e350b73f5ce4bc18c005f6dff9","analysis/generation-terminal-system-claim-guard-v1"),
"quality":("/tmp/cofitok-quality-bridge-execution-cf0e5fa/CoFiTok-internal","cf0e5faa94bf4ab38d947b921935b3b765b5537a","6cef27723196fd363379bca2e7b85b1678ebd777","scale/generation-stability-quality-bridge-100k"),
}
verified={}
for name,(cwd,rev,tree,branch) in GITS.items():
    observed=git(cwd)
    expected={"revision":rev,"tree":tree,"branch":branch,"dirty":False}
    if observed!=expected:
        raise RuntimeError(("git_mismatch",name,observed,expected))
    verified[name]=observed

standing=Path("/tmp/cofitok-quality-bridge-execution-cf0e5fa/standing_authorization.json")
if sha(standing)!="5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df":
    raise RuntimeError("standing_authorization_hash")
if json.load(open(standing)).get("status")!="active":
    raise RuntimeError("standing_authorization_inactive")
execution=json.load(open(R/"execution_status.json"))
controller_pid=int(execution["pid"])
if execution.get("status")!="running" or controller_pid<1 or not Path(f"/proc/{controller_pid}").is_dir():
    raise RuntimeError("controller_inactive")
if execution.get("full_training_launch_allowed") is not False or execution.get("full_300k_launch_allowed") is not False:
    raise RuntimeError("authorization_boundary_changed")
raw=Path(f"/proc/{controller_pid}/cmdline").read_bytes()
controller_argv=raw.rstrip(b"\0").split(b"\0")
if controller_argv!=[b"bash",b"artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh"]:
    raise RuntimeError(("controller_argv",controller_argv))
stat_tail=Path(f"/proc/{controller_pid}/stat").read_text().rsplit(")",1)[1].split()
controller={
    "pid":controller_pid,
    "start_ticks":int(stat_tail[19]),
    "executable":os.readlink(f"/proc/{controller_pid}/exe"),
    "cwd":os.readlink(f"/proc/{controller_pid}/cwd"),
    "cmdline_sha256":hashlib.sha256(raw).hexdigest(),
}

target_scripts=[
"run_generation_quality_bridge_controller_identity_guard.py",
"wait_generation_quality_bridge_runtime_compute_fairness.py",
"run_generation_quality_bridge_runtime_claim_guard_waiter.py",
"wait_for_generation_requested_class_visual_audit.py",
"wait_for_generation_terminal_distribution_support.py",
"wait_for_generation_quality_bridge_exposure_followup.py",
"wait_for_generation_training_exposure_audit.py",
"run_generation_factorization_quality_regression_supervisor.py",
"run_generation_matched_uncertainty_waiter.py",
"run_generation_quality_bridge_terminal_uncertainty_waiter.py",
"run_generation_quality_bridge_claim_qualification_waiter.py",
"run_generation_quality_bridge_claim_language_guard_waiter.py",
"run_generation_terminal_system_claim_guard_waiter.py",
]
for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        value=cmdline.read_bytes().replace(b"\0",b" ").decode(errors="replace")
    except OSError:
        continue
    if any(name in value for name in target_scripts):
        raise RuntimeError(("duplicate_process",cmdline.parent.name,value))

processes={}
def child_env(cwd,extra=None):
    result=os.environ.copy()
    result.update({
        "CUDA_VISIBLE_DEVICES":"-1",
        "OMP_NUM_THREADS":"1",
        "MKL_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1",
        "PYTHONPATH":str(cwd)+":"+str(cwd)+"/src",
    })
    if extra:
        result.update(extra)
    return result

def launch(name,cwd,argv,status_path,extra_env=None,accepted=("waiting","observing")):
    log_path=REC/(name+".log")
    handle=open(log_path,"ab",buffering=0)
    handle.write(("\n=== restart launch "+datetime.now(timezone.utc).isoformat()+" ===\n").encode())
    proc=subprocess.Popen(
        argv,cwd=str(cwd),env=child_env(cwd,extra_env),
        stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    deadline=time.time()+45
    latest=None
    while time.time()<deadline:
        if proc.poll() is not None:
            raise RuntimeError((name,"exited",proc.returncode,log_path.read_text(errors="replace")[-4000:]))
        try:
            latest=json.load(open(status_path))
            if int(latest.get("pid",-1))==proc.pid and latest.get("status") in accepted:
                break
        except (OSError,ValueError,json.JSONDecodeError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError((name,"no_fresh_status",latest))
    processes[name]={
        "pid":proc.pid,"cwd":str(cwd),"argv":argv,
        "status_path":str(status_path),"status":latest.get("status"),
        "detail":latest.get("detail"),"log":str(log_path),
    }
    return proc.pid

quality=Path(GITS["quality"][0])

controller_checkout=Path(GITS["controller"][0])
controller_dir=R/"controller_identity_guard_v1"
controller_receipt=controller_dir/("deployment_receipt."+STAMP+".json")
controller_binding=controller_dir/("controller_binding."+STAMP+".json")
launch("controller_identity_guard",controller_checkout,[
    PY,"scripts/run_generation_quality_bridge_controller_identity_guard.py",
    "--project",str(controller_checkout),
    "--quality-output-root",str(Q),
    "--execution-status",str(R/"execution_status.json"),
    "--quality-result",str(R/"quality_bridge_result.json"),
    "--pair-monitor",str(Q/"pair_monitor.json"),
    "--output-root",str(controller_dir),
    "--status-output",str(controller_dir/"guard_status.json"),
    "--pid-file",str(controller_dir/"guard.pid.json"),
    "--binding-output",str(controller_binding),
    "--deployment-receipt-output",str(controller_receipt),
    "--expected-controller-pid",str(controller["pid"]),
    "--expected-controller-start-ticks",str(controller["start_ticks"]),
    "--expected-controller-executable",controller["executable"],
    "--expected-controller-cwd",controller["cwd"],
    "--expected-controller-cmdline-sha256",controller["cmdline_sha256"],
    "--expected-control-revision",GITS["controller"][1],
    "--expected-control-tree",GITS["controller"][2],
    "--expected-control-branch",GITS["controller"][3],
    "--expected-training-revision",GITS["quality"][1],
    "--expected-training-branch",GITS["quality"][3],
    "--expected-monitor-name","generation_stability_full_data_quality_bridge_100k",
    "--required-identity-loss-polls","3",
    "--poll-seconds","60",
    "--timeout-seconds","2592000",
],controller_dir/"guard_status.json")

fairness=Path(GITS["fairness"][0])
fairness_dir=R/"runtime_compute_fairness"
fairness_receipt=fairness_dir/("deployment_receipt."+STAMP+".json")
fairness_pid=launch("runtime_compute_fairness",fairness,[
    PY,"scripts/wait_generation_quality_bridge_runtime_compute_fairness.py",
    "--project",str(fairness),
    "--launch-receipt",str(R/"launch_receipt.json"),
    "--expected-launch-receipt-sha256","4a9fd8577c24a92583d9538846dcab35c2d73e6066d164050f49b976985a3a21",
    "--config-validation",str(R/"config_validation.json"),
    "--expected-config-validation-sha256","cce5afbccac509903fceb93cf5bb3c7bcfa53a7637430545f6d0ae6b317b0fe2",
    "--runtime-selection",str(R/"runtime_selection.json"),
    "--expected-runtime-selection-sha256","a0d92f0db3cdcff5641bc94944b5b166952a915d4a693326df7a95c898d280fe",
    "--active-runbook",str(quality/"artifacts/runbooks/generation_stability_full_data_quality_bridge_100k_execute.sh"),
    "--expected-active-runbook-sha256","f0531763b4b888964a75923a8d59cb8cf8d1c79bd7782f845006c39e1a57fe73",
    "--cofitok-config",str(quality/"configs/generation/imagenet256_stability_quality_bridge_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"),
    "--dense-config",str(quality/"configs/generation/imagenet256_stability_quality_bridge_rollout_x0_u2_ema_teacher_dense_100k.json"),
    "--cofitok-run-dir",str(Q/"cofitok_rgbtail3_rollout_x0_u2_ema_teacher"),
    "--dense-run-dir",str(Q/"dense_rollout_x0_u2_ema_teacher"),
    "--cofitok-resume-compute-adjustment",str(R/"recovery_incident_2026-08-17_pin_memory/resume_compute_adjustment.json"),
    "--dense-resume-compute-adjustment",str(fairness_dir/"dense_identity/resume_compute_adjustment.json"),
    "--audit-output",str(fairness_dir/"final_report.json"),
    "--status-output",str(fairness_dir/"waiter_status.json"),
    "--deployment-receipt-output",str(fairness_receipt),
    "--expected-training-revision",GITS["quality"][1],
    "--expected-training-branch",GITS["quality"][3],
    "--expected-steps","100000",
    "--expected-auditor-revision",GITS["fairness"][1],
    "--expected-auditor-tree",GITS["fairness"][2],
    "--expected-auditor-branch",GITS["fairness"][3],
    "--expected-source-sha256","0caba8d7af93c83c8862e18cc1e77e0a7d497d4d8289173942fa5f2118efdf57",
    "--poll-seconds","60",
    "--timeout-seconds","2592000",
],fairness_dir/"waiter_status.json")

runtime=Path(GITS["runtime_claim"][0])
runtime_dir=R/"runtime_compute_claim_guard_v1"
runtime_receipt=runtime_dir/("deployment_receipt."+STAMP+".json")
runtime_pid=launch("runtime_claim_guard",runtime,[
    PY,"scripts/run_generation_quality_bridge_runtime_claim_guard_waiter.py",
    "--project",str(runtime),
    "--quality-output-root",str(Q),
    "--source-status",str(fairness_dir/"waiter_status.json"),
    "--source-final-report",str(fairness_dir/"final_report.json"),
    "--source-deployment-receipt",str(fairness_receipt),
    "--expected-source-deployment-receipt-sha256",sha(fairness_receipt),
    "--pair-monitor",str(Q/"pair_monitor.json"),
    "--output-root",str(runtime_dir),
    "--status-output",str(runtime_dir/"waiter_status.json"),
    "--pid-file",str(runtime_dir/"waiter.pid.json"),
    "--deployment-receipt-output",str(runtime_receipt),
    "--guard-output",str(runtime_dir/"runtime_compute_claim_guard.json"),
    "--expected-source-pid",str(fairness_pid),
    "--expected-control-revision",GITS["runtime_claim"][1],
    "--expected-control-tree",GITS["runtime_claim"][2],
    "--expected-control-branch",GITS["runtime_claim"][3],
    "--expected-source-control-revision",GITS["fairness"][1],
    "--expected-source-control-tree",GITS["fairness"][2],
    "--expected-source-control-branch",GITS["fairness"][3],
    "--expected-training-revision",GITS["quality"][1],
    "--expected-training-branch",GITS["quality"][3],
    "--expected-monitor-name","generation_stability_full_data_quality_bridge_100k",
    "--poll-seconds","60",
    "--timeout-seconds","2592000",
],runtime_dir/"waiter_status.json")

visual=Path(GITS["visual"][0])
launch("requested_class_visual_audit",visual,[
    PY,"scripts/wait_for_generation_requested_class_visual_audit.py",
    "--project",str(visual),
    "--quality-result",str(R/"quality_bridge_result.json"),
    "--cofitok-sampling-report",str(Q/"cofitok_rgbtail3_rollout_x0_u2_ema_teacher/terminal_100k/samples_10000_ddim100_cfg15/sampling_report.json"),
    "--dense-sampling-report",str(Q/"dense_rollout_x0_u2_ema_teacher/terminal_100k/samples_10000_ddim100_cfg15/sampling_report.json"),
    "--cofitok-dir",str(Q/"cofitok_rgbtail3_rollout_x0_u2_ema_teacher/terminal_100k/samples_10000_ddim100_cfg15/prefix_8"),
    "--dense-dir",str(Q/"dense_rollout_x0_u2_ema_teacher/terminal_100k/samples_10000_ddim100_cfg15/prefix_1"),
    "--classifier-calibration-report","/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_sampling_confirmation_10k_v1/diagnostics/class_fidelity_v2/real_validation_calibration_1perclass.json",
    "--real-dir","/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val",
    "--output-dir",str(R/"requested_class_visual_audit_terminal_100k_v1"),
    "--status",str(R/"requested_class_visual_audit_waiter_status.json"),
    "--lock",str(Q/"requested_class_visual_audit_waiter.lock"),
    "--python",PY,
    "--expected-revision",GITS["visual"][1],
    "--expected-tree",GITS["visual"][2],
    "--expected-branch",GITS["visual"][3],
    "--poll-seconds","60",
],R/"requested_class_visual_audit_waiter_status.json")

training_exposure=Path(GITS["training_exposure"][0])
launch("terminal_training_exposure",training_exposure,[
    "/usr/bin/bash",str(training_exposure/"artifacts/runbooks/generation_quality_bridge_terminal_training_exposure_waiter.sh"),
],R/"training_exposure_terminal_100k_waiter_status.json",{
    "PROJECT":str(training_exposure),"PYTHON":PY,
    "EXPECTED_SELF_REVISION":GITS["training_exposure"][1],
    "EXPECTED_SELF_TREE":GITS["training_exposure"][2],
    "EXPECTED_SELF_BRANCH":GITS["training_exposure"][3],
})

exposure=Path(GITS["exposure"][0])
exposure_waiter=exposure/"scripts/wait_for_generation_quality_bridge_exposure_followup.py"
launch("exposure_aware_followup",exposure,[
    "/usr/bin/bash",str(exposure/"artifacts/runbooks/generation_quality_bridge_exposure_followup_waiter.sh"),
],R/"exposure_aware_followup_waiter_status.json",{
    "PROJECT":str(exposure),"PYTHON":PY,
    "EXPECTED_REVISION":GITS["exposure"][1],
    "EXPECTED_TREE":GITS["exposure"][2],
    "EXPECTED_BRANCH":GITS["exposure"][3],
    "EXPECTED_WAITER_SHA256":sha(exposure_waiter),
    "QUALITY_BRIDGE_ROOT":str(Q),"POLL_SECONDS":"60",
})

distribution=Path(GITS["distribution"][0])
distribution_waiter=distribution/"scripts/wait_for_generation_terminal_distribution_support.py"
launch("terminal_distribution_support",distribution,[
    "/usr/bin/bash",str(distribution/"artifacts/runbooks/generation_quality_bridge_terminal_distribution_support_waiter.sh"),
],R/"terminal_distribution_support_v1/waiter_status.json",{
    "PROJECT":str(distribution),"PYTHON":PY,
    "EXPECTED_REVISION":GITS["distribution"][1],
    "EXPECTED_TREE":GITS["distribution"][2],
    "EXPECTED_BRANCH":GITS["distribution"][3],
    "EXPECTED_WAITER_SHA256":sha(distribution_waiter),
    "EXPECTED_FOLLOWUP_REVISION":GITS["exposure"][1],
    "EXPECTED_FOLLOWUP_BRANCH":GITS["exposure"][3],
    "QUALITY_BRIDGE_ROOT":str(Q),"POLL_SECONDS":"60",
    "NEAREST_CHUNK_SIZE":"128",
})

preceding=Path(GITS["preceding_uncertainty"][0])
evaluator=Path(GITS["evaluator"][0])
preceding_output=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_scaling_50k_matched_uncertainty_v1")
formal=evaluator/"configs/generation/diagnostics/matched_uncertainty_formal_10k_execution_v1.json"
confirmation=evaluator/"configs/generation/diagnostics/matched_uncertainty_confirmation_10k_execution_v1.json"
cache=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation/eval_cache/torch_fidelity/imagenet256_val_50k_torch_fidelity_v04__cofitok_19ace4e37bee2fca-inception-v3-compat-features-2048.pt")
preceding_pid=launch("matched_uncertainty_preceding",preceding,[
    PY,"scripts/run_generation_matched_uncertainty_waiter.py",
    "--project",str(preceding),
    "--evaluator-project",str(evaluator),
    "--quality-project",str(quality),
    "--quality-output-root",str(Q),
    "--output-root",str(preceding_output),
    "--status-output",str(preceding_output/"reports/waiter_status.json"),
    "--pid-file",str(preceding_output/"reports/waiter.pid"),
    "--formal-manifest",str(formal),
    "--confirmation-manifest",str(confirmation),
    "--expected-formal-manifest-sha256","2561b76113eb81c4288694302cc7fd683b806fad6fb420c2a0f1336b0ea3faac",
    "--expected-confirmation-manifest-sha256","e13518e421970fc7c2fe9ce12d22cb6636f77b421ba5af1b355e5623692a510a",
    "--real-feature-cache-source",str(cache),
    "--expected-real-feature-cache-bytes","409601577",
    "--expected-real-feature-cache-sha256","20103588dca9ce47bfceef6b68b473fdf4be720f149d1b8bdd96341d27c10dcd",
    "--python-executable",PY,
    "--expected-control-revision",GITS["preceding_uncertainty"][1],
    "--expected-control-tree",GITS["preceding_uncertainty"][2],
    "--expected-control-branch",GITS["preceding_uncertainty"][3],
    "--expected-evaluator-revision",GITS["evaluator"][1],
    "--expected-evaluator-tree",GITS["evaluator"][2],
    "--expected-evaluator-branch","",
    "--expected-quality-revision",GITS["quality"][1],
    "--expected-quality-tree",GITS["quality"][2],
    "--expected-quality-branch",GITS["quality"][3],
    "--poll-seconds","60","--required-idle-polls","5",
    "--timeout-seconds","2592000",
],preceding_output/"reports/waiter_status.json")

terminal_uncertainty=Path(GITS["terminal_uncertainty"][0])
terminal_output=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_terminal_matched_uncertainty_v1")
terminal_pid=launch("terminal_matched_uncertainty",terminal_uncertainty,[
    PY,"scripts/run_generation_quality_bridge_terminal_uncertainty_waiter.py",
    "--project",str(terminal_uncertainty),
    "--evaluator-project",str(evaluator),
    "--quality-project",str(quality),
    "--quality-output-root",str(Q),
    "--preceding-output-root",str(preceding_output),
    "--expected-preceding-pid",str(preceding_pid),
    "--expected-preceding-control-revision",GITS["preceding_uncertainty"][1],
    "--expected-preceding-control-tree",GITS["preceding_uncertainty"][2],
    "--expected-preceding-control-branch",GITS["preceding_uncertainty"][3],
    "--output-root",str(terminal_output),
    "--status-output",str(terminal_output/"reports/waiter_status.json"),
    "--pid-file",str(terminal_output/"reports/waiter.pid"),
    "--manifest-output",str(terminal_output/"reports/terminal_100k_execution_manifest.json"),
    "--real-feature-cache-source",str(cache),
    "--expected-real-feature-cache-bytes","409601577",
    "--expected-real-feature-cache-sha256","20103588dca9ce47bfceef6b68b473fdf4be720f149d1b8bdd96341d27c10dcd",
    "--python-executable",PY,
    "--expected-control-revision",GITS["terminal_uncertainty"][1],
    "--expected-control-tree",GITS["terminal_uncertainty"][2],
    "--expected-control-branch",GITS["terminal_uncertainty"][3],
    "--expected-evaluator-revision",GITS["evaluator"][1],
    "--expected-evaluator-tree",GITS["evaluator"][2],
    "--expected-evaluator-branch","",
    "--expected-quality-revision",GITS["quality"][1],
    "--expected-quality-tree",GITS["quality"][2],
    "--expected-quality-branch",GITS["quality"][3],
    "--poll-seconds","60","--required-idle-polls","5",
    "--timeout-seconds","2592000",
],terminal_output/"reports/waiter_status.json")

claim=Path(GITS["claim_qualification"][0])
claim_output=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_terminal_claim_qualification_v1")
claim_pid=launch("claim_qualification",claim,[
    PY,"scripts/run_generation_quality_bridge_claim_qualification_waiter.py",
    "--project",str(claim),
    "--terminal-output-root",str(terminal_output),
    "--terminal-status",str(terminal_output/"reports/waiter_status.json"),
    "--terminal-pid-file",str(terminal_output/"reports/waiter.pid"),
    "--quality-result",str(R/"quality_bridge_result.json"),
    "--execution-manifest",str(terminal_output/"reports/terminal_100k_execution_manifest.json"),
    "--uncertainty-report",str(terminal_output/"terminal_100k/matched_uncertainty.json"),
    "--output-root",str(claim_output),
    "--status-output",str(claim_output/"reports/waiter_status.json"),
    "--pid-file",str(claim_output/"reports/waiter.pid"),
    "--qualification-output",str(claim_output/"reports/statistical_claim_qualification.json"),
    "--expected-terminal-pid",str(terminal_pid),
    "--expected-control-revision",GITS["claim_qualification"][1],
    "--expected-control-tree",GITS["claim_qualification"][2],
    "--expected-control-branch",GITS["claim_qualification"][3],
    "--expected-terminal-control-revision",GITS["terminal_uncertainty"][1],
    "--expected-terminal-control-tree",GITS["terminal_uncertainty"][2],
    "--expected-terminal-control-branch",GITS["terminal_uncertainty"][3],
    "--expected-quality-revision",GITS["quality"][1],
    "--expected-quality-tree",GITS["quality"][2],
    "--expected-quality-branch",GITS["quality"][3],
    "--expected-evaluator-revision",GITS["evaluator"][1],
    "--expected-evaluator-tree",GITS["evaluator"][2],
    "--poll-seconds","60","--timeout-seconds","2592000",
],claim_output/"reports/waiter_status.json")

language=Path(GITS["claim_language"][0])
language_output=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_terminal_claim_language_guard_v1")
launch("claim_language_guard",language,[
    PY,"scripts/run_generation_quality_bridge_claim_language_guard_waiter.py",
    "--project",str(language),
    "--source-output-root",str(claim_output),
    "--source-status",str(claim_output/"reports/waiter_status.json"),
    "--source-pid-file",str(claim_output/"reports/waiter.pid"),
    "--source-report",str(claim_output/"reports/statistical_claim_qualification.json"),
    "--output-root",str(language_output),
    "--status-output",str(language_output/"reports/waiter_status.json"),
    "--pid-file",str(language_output/"reports/waiter.pid"),
    "--guard-output",str(language_output/"reports/statistical_claim_language_guard.json"),
    "--expected-source-pid",str(claim_pid),
    "--expected-control-revision",GITS["claim_language"][1],
    "--expected-control-tree",GITS["claim_language"][2],
    "--expected-control-branch",GITS["claim_language"][3],
    "--expected-source-control-revision",GITS["claim_qualification"][1],
    "--expected-source-control-tree",GITS["claim_qualification"][2],
    "--expected-source-control-branch",GITS["claim_qualification"][3],
    "--poll-seconds","60","--timeout-seconds","2592000",
],language_output/"reports/waiter_status.json")

terminal_system=Path(GITS["terminal_system"][0])
terminal_system_dir=R/"terminal_system_claim_guard_v1"
launch("terminal_system_claim_guard",terminal_system,[
    PY,"scripts/run_generation_terminal_system_claim_guard_waiter.py",
    "--project",str(terminal_system),
    "--quality-output-root",str(Q),
    "--quality-result",str(R/"quality_bridge_result.json"),
    "--statistical-claim-guard",str(language_output/"reports/statistical_claim_language_guard.json"),
    "--statistical-waiter-status",str(language_output/"reports/waiter_status.json"),
    "--visual-audit-waiter-status",str(R/"requested_class_visual_audit_waiter_status.json"),
    "--runtime-claim-guard",str(runtime_dir/"runtime_compute_claim_guard.json"),
    "--runtime-waiter-status",str(runtime_dir/"waiter_status.json"),
    "--output",str(terminal_system_dir/"terminal_system_claim_guard.json"),
    "--status-output",str(terminal_system_dir/"waiter_status.json"),
    "--lock",str(terminal_system_dir/"waiter.lock"),
    "--expected-revision",GITS["terminal_system"][1],
    "--expected-tree",GITS["terminal_system"][2],
    "--expected-branch",GITS["terminal_system"][3],
    "--poll-seconds","60","--timeout-seconds","2592000",
],terminal_system_dir/"waiter_status.json")

factor=Path(GITS["factor"][0])
factor_dir=R/"factorization_quality_regression_supervisor_v1"
factor_receipt=factor_dir/("deployment_receipt."+STAMP+".json")
launch("factorization_quality_regression_supervisor",factor,[
    PY,"scripts/run_generation_factorization_quality_regression_supervisor.py",
    "--project",str(factor),
    "--expected-revision",GITS["factor"][1],
    "--expected-tree",GITS["factor"][2],
    "--expected-branch",GITS["factor"][3],
    "--preparation",str(factor_dir/"preparation.json"),
    "--expected-preparation-sha256","1b5696a30a3a13da5cb3e3552951531798d7711e324d9a262196b08f269bc724",
    "--standing-authorization",str(standing),
    "--expected-standing-authorization-sha256","5fe64a0941acb55a21cb6479a726987c6259e93a772c47290f2d6883752de4df",
    "--followup-decision",str(R/"followup_experiment_decision_exposure_aware_v2.json"),
    "--terminal-system-guard",str(terminal_system_dir/"terminal_system_claim_guard.json"),
    "--source-binding",str(factor_dir/"source_binding.json"),
    "--execution-authorization",str(factor_dir/"execution_authorization.json"),
    "--output-root","/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_factorization_quality_regression_v1",
    "--runbook",str(factor/"artifacts/runbooks/generation_factorization_quality_regression_probe_v1.sh"),
    "--python",PY,
    "--status-output",str(factor_dir/"supervisor_status.json"),
    "--pid-file",str(factor_dir/"supervisor.pid.json"),
    "--deployment-receipt-output",str(factor_receipt),
    "--poll-seconds","60","--required-idle-polls","5",
    "--timeout-seconds","2592000",
],factor_dir/"supervisor_status.json")

receipt={
    "schema_version":1,
    "role":"generation_quality_bridge_server_restart_downstream_waiter_recovery",
    "status":"pass",
    "created_at":datetime.now(timezone.utc).isoformat(),
    "hostname":socket.gethostname(),
    "quality_bridge_root":str(Q),
    "controller":controller,
    "quality_git":verified["quality"],
    "standing_authorization":{"path":str(standing),"sha256":sha(standing)},
    "scope":{
        "current_quality_bridge_only":True,
        "cpu_only_supervisors":True,
        "duplicate_gpu_work_launched":False,
        "capacity_pipeline_restored":False,
        "full_training_launch_allowed":False,
        "full_300k_launch_allowed":False,
        "promotion_or_release_allowed":False,
    },
    "processes":processes,
}
tmp=REC/"recovery_receipt.json.tmp"
tmp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
os.replace(tmp,REC/"recovery_receipt.json")
print(json.dumps({
    "status":"pass",
    "receipt":str(REC/"recovery_receipt.json"),
    "pids":{name:data["pid"] for name,data in processes.items()},
},indent=2))

