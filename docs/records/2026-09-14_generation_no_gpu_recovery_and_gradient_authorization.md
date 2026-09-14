# 2026-09-14：无卡恢复、证据现状与梯度诊断授权

截至 2026-09-14 19:00 CST，SSH 已恢复。缺失的锁定代码副本已恢复，
四臂终端 SNR screen 的完整物理复验通过；用户此前回复的“授权”已落盘，
原实现生成了执行授权，独立验证器的 admission replay 通过。
**梯度诊断尚未运行，结果目录不存在，完成测量数为 0/256。**
当前限制是无卡运行环境；后续不需要重复征求同一诊断的授权。

## 已有成果与尚未支持的主张

| 实验／问题 | 已核实结果 | 能支持的结论 |
| --- | --- | --- |
| 锁定的 ImageNet-256 K4、20K、双种子机制实验 | 1,024 张验证图、固定 t=500；7/7 gate 通过；两个种子的 learned order 都在 24 个排列中排名第 1；endpoint MSE 对 direct dense 平均降低 6.73% | ordered restricted noise factorization 与 prefix-controllable denoising；不是正式生成质量优势 |
| 锁定的八数据集 short-budget 实验 | path AUC 对 endpoint-only 为 8/8 胜；生成 lowres 最优 0/8、Inception-style 最优 1/8 | 存在机制证据；不能写 broad generation SOTA |
| 四臂 terminal-SNR screen，每臂 10K 训练与 1K 样本 | 19 项检查中 17 项通过；两种方法的相对 FID 改善分别为 -11.3635% / -9.4088%，均未达到至少 +5%；四臂 recall 都为 0 | 所测终端 SNR 改动没有修复生成质量，结果保持 hold |
| frozen 100K checkpoint 对应的 20K 既有图像分类诊断 | CoFiTok/dense top-1 为 0.24% / 0.17%，AMI 为 0.003919647 / 0.003033364；held-out relabel 为 0.14% / 0.09% | 类别关联弱且支持集集中，未证实重映射可以修复；不能说条件完全被忽略或共同训练原因已确定 |

早期正面证据来源：
`artifacts/reports/imagenet256_confirmatory_2026-07-11/imagenet256_confirmatory_report.md`
和 `artifacts/reports/paper_evidence_report_2026-07-11_final/paper_evidence_report.md`。
分类诊断来源：`docs/records/2026-09-13_generation_class_support_contingency_result.md`。
100K 既有样本的 recall 0.00832 / 0.01000 与四臂 10K screen 的 recall=0 是不同实验，不能混用。

因此，已有实验成果支持受限范围内的机制主张，但生成系统升级迄今没有成功。
代码测试、证据复验和授权准备属于工程工作，不计为新的科学实验或生成质量突破。

## 本轮物理复验

权威 screen 根目录：
`/root/autodl-tmp/CoFiTok/checkpoints/generation/terminal_snr_endpoint_screen_v1`。
controller 仍为 `completed / complete / hold`，无 screen/controller/evaluator 进程。
六个 prelaunch 文件均与用户固定的 SHA256 一致。

轻量标准库审计核验四臂的 45 个关联来源文件、全部 5K/10K checkpoint 与 sidecar、
latest/report、201 行严格递增且数值有限的 metrics，以及全部 4,000 张 PNG 的原生 sample-set
digest。每臂末步均为 10,000，`samples_seen=step*64`，sampling progress 均为 completed。

此 screen 按协议在 10K 截止，配置 target 为 100K；报告中的
`completed_steps=10000 / target_steps=100000 / training_complete=false`、
`stop_requested=false / stop_signal=null` 是正确值。最初临时摘要检查误要求
`training_complete=true`，已修正该临时检查；未修改报告、训练器或锁定验证器。

恢复 exact clean validator 后，重新调用原实现的四个 arm replay、result replay 和
adjacent receipt validation，全部完成且进程退出 0。此前本轮只输出三个臂后退出的
不完整尝试不算完整 pass；本次完整复验取代其当前验证状态，不覆盖原证据。
所有 19 项科学检查已核对，仅两个 relative-FID-improvement 检查失败。
recall regression 为 0 的通过只表示两边同为零，没有证明分布覆盖改善。

以下身份保持不变：

| 文件 | SHA256 |
| --- | --- |
| controller_status.json | `dfba8a85c6aa0b530e27874c5a0187f18ace3ac2624ab0cfedd92b8f4dd45e14` |
| controller.log | `2167f99c0c899c7e79896dc2924849e9a5bd8638e9c985777f13f97edaca084d` |
| terminal_snr_screen_result.json | `b740c8b21aabf640c156aea076058c73350d26c110e2cc967576a26492a52dea` |
| terminal_snr_screen_result.validation.json | `3be043d852bd61d6e79005dc9d14f77113f1250bef967ca4e2a8ac74bc715f95` |

## 恢复缺失的执行代码

18:50 CST 的实时检查发现七个相关 `/tmp` checkout 缺失；数据盘上的权重、结果、准备与
用户授权文件仍存在且哈希一致。此处只记录观察，不推断缺失原因。
从本地 Git 构建了恢复包，服务器在 fetch 前通过 prerequisite 验证：

```text
/root/autodl-tmp/CoFiTok/checkpoints/generation/.code_recovery_2026-09-14/
  cofitok-code-recovery-cfd1296-a5199e6-from-1ebcc15.bundle
bytes: 40680477
sha256: d60b380b66564445cb392b2b125523e94adb1ed7a478ad3a1e06d5485d2a96ad
required commits:
  1ebcc15210e63a776a2ba448481cbd8bb94a4066
  58d83bfce2770eab2565b8c89a5f9a06201a0c86
advertised commits:
  cfd129608f399b13cb7ac1f8405001e9ffe24ef6
  a5199e6eea63817fe4eee59ac187766748d6e772
```

该包用于恢复代码，不是新的实验执行包，也不替代原 screen/confirmation bundle 的身份。
fetch 仅发生在同目录下新建的独立 `source.git`，保留在数据盘以便后续恢复。
正式主仓库始终为 `1ebcc15210e63a776a2ba448481cbd8bb94a4066`、tracked clean，未被部署或移动 HEAD。

恢复后的目录、commit/tree/branch 均精确匹配已绑定身份：

- screen execution：`/tmp/cofitok-terminal-snr-execution-89bcd9a`，commit `89bcd9a...`。
- screen validator：`/tmp/cofitok-terminal-snr-confirmation-rehearsal-a5199e6-v1`，commit `a5199e6...`。
- gradient preparation/authorization/evaluator/validator：分别为
  `/tmp/cofitok-frozen-gradient-<role>-ad237ef`，commit `ad237efa810df8677b4187110a723279d47f9d4b`、
  tree `e5b0def15e93df13b845abbc905cff7eda7bc486`。
  四份 checkout 各自 89 个已绑定 Python 文件全部通过 SHA256 比较。
- catalog builder：`/tmp/cofitok-frozen-loss-gradient-preparation-5ac6192`，commit
  `5ac619228b051d12d87a2d790a3676d31c8d7b72`、tree `693704cc227e7ac213be3686a920214556525058`。

恢复不覆盖任何已有目录，不更改 screen 输出、controller state、模型、配置、目标或运行协议。
历史 exposure 目录和 locked paper evidence 未被改写。

## 已落盘并独立复验的授权

控制目录：
`/root/autodl-tmp/CoFiTok/checkpoints/generation/.frozen_loss_gradient_attribution_v1.control/`。

| 文件 | Bytes | SHA256 |
| --- | ---: | --- |
| preparation.json | 51660 | `8cc706a237812d137389b44985c599789bbca38d3cfa02e8ceb8c07c26cd422b` |
| stage_approval.json | 13619 | `5631879393eaa9eb010bdaf72b39a22050fbc224f93e12033a2db168d06b9060` |
| execution_authorization.json | 13316 | `24dab51374333fe4f6b271fd48a1a3f3e1c012c4b55ba475e93745d982d4fca2` |

三份文件均为 0444。用户明确回复“授权”，Codex 作为记录者将其绑定到已展示的准备与
诊断范围；`approved_at_utc=2026-09-14T10:40:50Z` 明确是记录时间，不冒充用户消息的精确时间戳。
准备文件 mtime_ns 保持 `1789233150586218883`，用户授权文件保持 `1789382629335289413`。

较早的一次授权构建子进程退出 -9，没有 Python traceback；当时 cgroup OOM 计数为零，
不能把原因断言为 OOM。恢复代码并完成串行 screen replay 后，直接运行原 exact authorization
checkout 的 `authorize` CLI 成功、退出 0。独立 exact validator 随后重放 preparation/catalog、
两份原始 100K checkpoint 来源、32 张固定图像及授权绑定，`admission(..., role="validator")`
通过、退出 0。没有另写替代授权算法，没有改变原执行器。

授权范围仍为 32 张图 × 4 个 timestep × 2 方法，共 256 个单图梯度测量，最多一次、7,200 秒。
使用冻结的 100K online 权重和原 EMA teacher，CUDA/bf16，无 optimizer/EMA 更新、无训练、无新采样。
独立单位是 32 张图；这不是原 batch-64 的梯度重放，也不是因果或质量改善证明。

本地小型档案位于
`artifacts/reports/generation/frozen_gradient_authorization_2026-09-14/`，包括字节一致的两份授权文件、
恢复前/后实时快照和独立 admission 结果。快照中纳秒时间戳用十进制字符串保存，避免 JSON
跨运行时的大整数精度损失。机器观察记录不是新的 scientific validation receipt。

## 当前硬件与下一步

最终检查时间 2026-09-14T11:00:07.681805+00:00：没有 `/dev/nvidia*`，`nvidia-smi`
不可用，实例内存上限 `2147483648` bytes（2 GiB），磁盘可用 `189437259776` bytes。
进程树只有平台服务与本次只读检查，没有训练、采样或梯度 evaluator；本轮未 signal/kill 任何进程。
CPU 验证时设置的单线程、禁用 CUDA 环境变量仅作用于对应命令，未写入环境配置，不能带入真实 assay。

恢复原 RTX PRO 6000 和原定运行环境后，使用上述现有授权继续：先重核所有来源、GPU PID ownership、
内存/磁盘与冻结 runtime，确认输出根仍不存在；随后只启动一次 exact evaluator `run`。
结束后由独立 exact validator 校验全部 256 行、标量重算、tensor preservation 与来源绑定，并保留
成功或失败的所有证据。不能绕过 runtime 检查改成 CPU，也不能自动重试已占用的输出目录。

当前未调用 `run`；
`/root/autodl-tmp/CoFiTok/checkpoints/generation/frozen_loss_gradient_attribution_v1` 不存在。
confirmation、full-training/300K、promotion、export、release、paper integration 权限全部仍为 false。
完整生成系统目标尚未完成。
