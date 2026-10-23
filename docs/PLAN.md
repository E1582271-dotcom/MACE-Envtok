# EnvTok → MACE：从零实现并 8 卡分布式训练一套 MLIP（复现 arXiv 2610.14392），最终推送 GitHub

## Context

- 目标：复现 Moerman et al.《Atoms that Matter》(arXiv 2610.14392) 的 **EnvTok 子集模型**（OMat24 中挑出的 795,721 个结构、约 657 万原子，不到全集 1%），做成可公开的 GitHub 仓库，用作简历项目。项目要证明你能从零写出 MLIP 模型、PyTorch 训练栈和多机 DDP 分布式训练。
- 已定决策（2026-10-21）：
  - 算力：NSCC ASPIRE 2A 个人项目额度，1,562.5 A100·h，一次性额度。
  - 模型和训练栈都自己写。球谐、CG、张量积用 e3nn / cuEquivariance 原语，当黑箱用。mace-torch 只作对拍的标准答案。
  - **主训练用 8 卡：2 节点 × 4 A100，跨节点 DDP**。配置与作者一致：每卡 128，全局 batch 1024。
  - **不设人为时间或 GPU·h 上限**，靠早停（patience 40，与作者一致）结束。
  - 只训 EnvTok。DIRECT / Random 对比和随机种子方差都用作者公开的模型（DIRECT、rdm1–4）及其已发布的基准结果。
  - 新方法（novel method）先不做，主结果出来再定。
  - **OOD 明确化**：评测结果按 ID（分布内）/ OOD（分布外）组织，另加潜空间距离分析，研究“离训练集越远误差越大”（M5，约 15 GPU·h）。不确定度估计不做，留给 M7。
- 现状：`~/Desktop/EnvTok-MACE/` 是空目录，从零开始。
- 时间预期：作者“几天训完”只是训练本身，而且用的是成熟库。我们最快约 **8 周（12 月中旬）**出最终数字并推送；8 卡训练本身约 3–5 天墙钟。M1 对拍通过后（约 11 月中）就可以先写“重新实现”那条 bullet。

## 关键事实（已核实）

- **数据**
  - Zenodo 10.5281/zenodo.23137852 提供四类文件：
    - `subsets.zip`：EnvTok / DIRECT / rdm1–4 的 ID 列表；
    - `models.zip`：作者训练好的模型；
    - `benchmarks_*.zip`：MatPES / MatCalc / MD 稳定性 / nebDFT2k 的结果；
    - 许可证为 CC BY 4.0。
  - 原始结构要从 OMat24 下载：11 个 train 子集，格式 `.aselmdb` tar.gz。
  - 剔除稀土后的元素表为 {1..56} ∪ {72..83}，共 68 个元素。
  - E0s 用作者 `preprocess.yaml` 里的 OMat24 PBE 孤立原子能量，固定不拟合。
- **超参**（作者 `07_training/train.yaml`，mace-torch v0.3.14 / torch 2.5.0 / e3nn 0.4.4）
  - 网络：第 1 层 `RealAgnosticDensityInteractionBlock`（无残差），第 2 层 `RealAgnosticDensityResidualInteractionBlock`。hidden 128x0e+128x1o，max_ell 3，correlation 3，10 个 Bessel 基，r_max 6.0。
  - 两个附加项：Agnesi 距离变换（参数固定）和 ZBL 近程排斥。
  - 损失：UniversalLoss，E/F/S 权重 1/10/10，Huber δ=0.01。
  - 优化：Adam + AMSGrad，L2 权重衰减 1e-6，只作用于 `products.*` 和 interaction 里的 `linear.weight`。lr 1e-3，梯度裁剪 100，EMA 0.999（带 warmup）。
  - 调度与批量：ReduceLROnPlateau（patience 5，factor 0.8），每卡 128 × 8 卡 = 1024，fp32，早停 patience 40。
- **作者成本**：同样是 2 节点 × 4 A100、开 cueq 加速，832 epoch 用时约 3.8 天，约 730 GPU·h，平均约 6.6 分钟一个 epoch（其中约 30% 花在每个 epoch 都跑的全量 val）。最佳 epoch 是 792。
- **NSCC ASPIRE 2A**
  - 节点：每节点 4×A100-40G，节点内 NVLink；节点间是 HPE Slingshot（网卡 hsn0），不是 InfiniBand。
  - 调度与配额：PBS Pro；GPU 队列 g1–g4 最长 24 h，glong 1–64 卡、最长 120 h；每卡自动配 16 核和约 110 GB 内存。
  - 环境与存储：计算节点离线；Singularity 要加 `--nv`；home 50 GB；scratch 30 天不动就清。
- **论文对比数字**
  - MatPES bulk 力 MAE：子集模型 149–151 meV/Å，MACE-OMAT-0 为 126。
  - 其余对比项：Table 1 分箱、生成能 MAE 63–74 meV/atom、MD 完成率 EnvTok 92%、Li⁺ 迁移势垒 MAE（Table 2）。

## 8 卡训练策略

### 1. 核心判断：瓶颈在哪
- 模型约几百万参数，每步梯度只有几十 MB。即使跨节点只走普通 socket，all-reduce 也只要几十毫秒，而一步计算约 0.5 s 以上。所以**跨节点带宽不是瓶颈**。
- 8 卡效率主要取决于两点：
  - **掉队 rank**：每个结构的原子数不同，DDP 每一步都要等最慢的那张卡；
  - **数据供给**：CPU 读数据跟不上 GPU。
- 策略的重点因此放在负载均衡和数据加载上。通信只要确认走对网卡、不挂死即可。

### 2. 资源申请与队列
- 每个作业 `select=2:ngpus=4`（2 个 chunk，每节点 4 卡），CPU 和内存按每卡 16 核 / 约 110 GB 配齐。具体字段以 NSCC 官方 GPU 作业模板为准，M0 用 `qstat -Q` 核对队列名和上限。
- 试跑（M3）走 24 h 的 g 队列，每个作业 2–4 h，排队短。
- 正式训练走 glong，单作业申请 120 h，同时预先挂一个 `-W depend=afterany` 的接续作业。前一个作业已经早停结束时，接续作业读到 `FINISHED` 标记文件就立即退出，只耗约 1 分钟。

### 3. 进程模型与启动
- 1 GPU = 1 进程，共 8 个 rank：rank = node_rank × 4 + local_rank，每个进程只看到自己那张卡。
- **主方案**：`mpiexec -n 8 --ppn 4` 直接起 8 个 Python 进程。rank、local_rank、world_size 从 MPI 环境变量读取，Cray PALS 或 OpenMPI 的变量名都兼容，M0 确认具体实现。MASTER_ADDR 取 `$PBS_NODEFILE` 的第一台主机，然后 `init_process_group("nccl")`。
- **备选方案**：`mpiexec` 每节点起 1 个 `torchrun --nnodes 2 --nproc_per_node 4 --rdzv_backend c10d`。两种都在 M3 测一遍，用更稳的那种。
- NCCL 环境变量：
  - `NCCL_SOCKET_IFNAME=hsn0`；
  - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`；
  - 进程组超时设 30 分钟，挂死的作业会报错退出，不会空烧 GPU·h；
  - 试跑时加 `NCCL_DEBUG=INFO`，确认走的是 Slingshot（OFI 插件，aws-ofi-nccl）还是 socket。两者都能用，记录实测带宽。

### 4. 数据切分、负载均衡与加载
- **加载**：每个 rank 启动时把训练集 HDF5（约 3 GB）整体读进内存，不走 GPFS 随机读。val 子集也一样。每节点 4 个 rank 共约 12 GB，节点内存充足。
- **切分**：自写 `BalancedDistributedSampler`：
  1. 每个 epoch 用固定种子和 epoch 号打乱全集，切成 777 个全局 batch，每个 1024 个结构，`drop_last`；
  2. 在每个全局 batch 内按边数排序，再“蛇形”分给 8 个 rank，每 rank 128 个结构，边数大致相等。
- 这样**全局 batch 的成员与随机采样完全相同**，只是在 rank 之间重排。配合全局归一化，梯度在数学上不变，但每步等待最慢 rank 的时间更短。M3 对比开启与关闭这个 sampler 时，每步的最大/平均 rank 耗时比。
- DataLoader：每 rank 8–12 个 worker（每卡有 16 核），`pin_memory`、`persistent_workers`、`prefetch_factor=4`。数据从内存数组读，不需要在 worker 里开 HDF5。

### 5. 梯度同步与优化器（保证 8 卡 = 单卡 1024）
- DDP 配置：
  - `DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)`；buffer 都是常量，不必广播。
  - 不开 `find_unused_parameters`，所有参数每步都有梯度，按元素索引的权重用 one-hot 乘法实现，保证这一点。
  - 力用 `autograd.grad(create_graph=True)` 在 forward 内对输入求导，DDP 的 hook 只在 `loss.backward()` 时触发一次。all-reduce 默认与反传重叠。
- **全局归一化**：E/F/S 三项先 all_reduce 计数，loss = 本地误差和 × world / 全局计数。DDP 求平均后，梯度等于全局 1024 个结构上的均值。这样 8 卡的梯度和单卡（累积 8 步）的梯度相同。
- 学习率保持 1e-3：全局 batch 和作者一样，不需要线性缩放。
- 梯度裁剪（100）在同步后的梯度上做，各 rank 结果一致。
- EMA 在每个 rank 上同步更新，参数相同所以结果相同。保存时只取 rank 0。
- 初始化固定种子，DDP 构造时也会从 rank 0 广播参数。
- 精度用 fp32，与作者一致，默认不开 TF32。

### 6. 验证、调度器与早停
- 固定的分层 10% val 子集（约 6.6 万帧），切成 8 份，用不补齐的自写 sampler，保证指标精确。用 EMA 权重算，误差和与计数在 rank 间 all_reduce。
- ReduceLROnPlateau 在每个 rank 上用同一个 all-reduce 后的 val loss 更新，所以 lr 一致。
- 早停由 rank 0 判断，再用 broadcast 把“停止”标志发给所有 rank，8 个进程同时退出。作者的版本只有 rank 0 break，导致 DDP 报错。
- 全量 val 只在最佳 checkpoint 上跑。

### 7. Checkpoint 与容错
- 只由 rank 0 写 checkpoint：各 rank 的 RNG 状态先 gather 到 rank 0，和模型、EMA、optimizer、scheduler、早停状态、epoch/step、sampler 种子一起，先写临时文件再 rename（原子写），前后加 barrier。
- 频率：
  - 每个 epoch 覆盖一次 `last`；
  - val 改善时存 `best`，同时拷回 home；
  - 每 25 epoch 存一份留档。
  - 全程磁盘占用可控，作者存了每个 epoch，约 90 GB。
- **walltime 保护**：训练器知道作业结束时间，每步检查。剩余时间少于 max(20 分钟, 2 个 epoch) 时，保存 `last` 后正常退出，接续作业自动从 `last` 续训。
- 节点故障或 NCCL 超时会让作业报错退出，接续作业照样从 `last` 恢复，最多损失 1 个 epoch（几分钟）。
- **NaN 保护**：loss 非有限时，各 rank 通过 all-reduce 一致跳过这一步并计数。连续 20 步出现 NaN 就保存后停止，留待排查。

### 8. 监控
- rank 0 每 50 步写一行 CSV 并记 wandb offline，内容包括：
  - 吞吐（结构数/s、原子数/s）；
  - 每步耗时，以及 8 个 rank 耗时的最大/平均比（all_gather 得到，用来发现掉队）；
  - all-reduce 耗时、显存峰值、lr、loss。
- 每个 epoch 记录 val MAE 和剩余额度估算。
- `scripts/nscc/status.sh` 输出一屏摘要：作业状态、当前 epoch、最近 val、吞吐、预计剩余时间与 GPU·h。你在 NSCC 上运行后把输出贴给我，我来判断是否需要干预。

### 9. 性能目标与排障顺序（M3 验收）
- 目标：1→8 卡扩展效率 ≥ 85%（节点内 1→4 卡 ≥ 90%）；平均每 epoch 时间不超过作者的约 1.5 倍；推算到 832 epoch 的 GPU·h 写进预算。
- 不达标时按以下顺序排查：
  1. 数据供给：看 GPU 利用率，数据加载等待时间占比；
  2. 掉队：看 rank 耗时比，确认均衡 sampler 生效；
  3. 后端：确认 cueq 已启用；
  4. NCCL：确认走 hsn0 / OFI；
  5. 小核函数开销：考虑每卡更大的 batch 并减少累积。

### 10. 8 卡正确性证明（简历与面试的硬证据）
- **梯度等价**：同一组 1024 个结构，单卡（累积 8 × 128）和 8 卡各算一次梯度，fp32 相对误差 < 1e-5。
- **曲线等价**：50k 子集上，8 卡和单卡累积各跑 200 步，loss 曲线重合。
- **与官方训练器对照**：同种子、同后端、同为 8 卡，自写训练器与 `mace_run_train` 的 loss 曲线基本重合。
- **续训等价**：8 卡上中断后续训，曲线连续，误差在容差内。GPU 上的 scatter_add 不确定，所以逐位一致只在 CPU 上测。
- **扩展效率曲线**：1/2/4/8 卡，节点内和跨节点分开，结果写进 `results/figures/scaling.png`。

### 11. 墙钟与预算估算
- cueq 后端下，若吞吐接近作者，每 epoch 约 5–7 分钟（用 10% val，比作者省掉约 30%），832 epoch ≈ 70–100 h ≈ 560–800 GPU·h，一个 glong 作业大概率够。
- 纯 e3nn 版本预计慢 2–4 倍，所以 cueq 后端是 M3 必做项。
- 这些都只是估算，M3 实测后更新。

## 文件夹管理（仓库根 = GitHub 仓库）

```
EnvTok-MACE/
├── README.md            # 简介、结果表、复现步骤、引用（M6 定稿）
├── LICENSE              # MIT
├── CITATION.cff
├── pyproject.toml  uv.lock   # 包名 envtok_mace，依赖版本锁定
├── .gitignore  .env.example  # .env（NSCC 项目号等）不入库
├── src/envtok_mace/
│   ├── data/    extract_subset.py · preprocess.py · dataset.py · samplers.py (均衡/不补齐 sampler) · collate.py
│   ├── model/   radial.py · zbl.py · blocks.py · mace.py · backends.py (e3nn / cueq 可切换)
│   ├── train/   loss.py · optim.py · ema.py · dist.py (初始化/全局归一化/广播) · checkpoint.py · trainer.py · tracking.py
│   ├── eval/    metrics.py · run_eval.py · ase_calc.py
│   ├── bench/   matpes.py · matcalc.py · md_stability.py · nebdft2k.py
│   ├── analysis/ latent.py (特征提取/PCA/kNN 距离) · id_ood.py (ID/OOD 汇总表与误差-距离曲线)
│   └── cli.py   # preprocess / train / eval / bench / analyze 五个子命令
├── configs/  model/omat0_medium.yaml · train/{debug_cpu,pilot_8gpu,envtok_8gpu}.yaml · bench/*.yaml
├── scripts/
│   ├── data/   download_zenodo.sh · download_omat24.sh · download_benchmarks.sh
│   └── nscc/   setup_env.sh · extract.pbs · train_8gpu.pbs · launch_8gpu.sh · chain.sh · status.sh · bench.pbs
│               # 项目号与路径一律从环境变量读，脚本里不写死任何账号
├── tests/    test_*.py + fixtures/（几十个 OMat24 结构，<1 MB）
├── docs/     PLAN.md（本计划去掉账号信息后落盘）· PARITY.md · NSCC.md · RESULTS.md
├── results/  只放小文件：tables/*.csv · figures/*.png|svg · runs/<run_id>/summary.json
└── local/    整个目录 gitignore
    ├── third_party/  mace v0.3.14、作者仓库 omat24-chemenv-subset 的克隆
    ├── zenodo/       subsets/ models/ benchmarks_*/
    ├── data/         本机调试用的小样本 HDF5
    ├── runs/         本机调试的 checkpoint 和日志
    └── notes/        私人笔记及任何含账号信息的内容
```

- **NSCC 端布局**：
  - `$HOME/EnvTok-MACE`：git clone 的代码；
  - `$HOME/envtok_data/processed`：训练集、val、MatPES 的 HDF5，约 15 GB，放 home 持久保存；
  - `$HOME/envtok_runs/<run_id>/best`：最佳 checkpoint 备份；
  - `$SCRATCH/envtok/raw`：OMat24 原始数据，抽取验证后删除；
  - `$SCRATCH/envtok/runs/<run_id>`：checkpoint、日志、wandb offline 记录。
- **可追溯性**：每个 run 目录自动写入 config 快照、git commit SHA、环境版本、节点列表、NCCL 传输方式。results/ 里的每个数字都能追溯到一个 run_id。
- **Git 流程**：
  - M0 当天在仓库根 `git init`，主分支 main，每完成一个可测的小步就 commit。
  - 里程碑打 tag：`m1-parity`、`m2-trainer`、`m3-8gpu`、`m4-trained`、`v1.0`。
  - GitHub 远程仓库建议先建 private，M6 发布检查通过后再改 public。每次 push 我都会先问你。
- **推送前检查**（M6）：
  - 账号与路径：无 NSCC 账号或项目号，无带用户名的绝对路径，`.env` 未入库。
  - 大文件：无数据、无 checkpoint，没有任何文件超过 50 MB。训练好的模型放 GitHub Release 或给 Zenodo 链接。
  - 内容：README 里的数字与 results/ 一致；`pytest` 全过；LICENSE 和引用齐全。
  - 身份：确认 `gh auth status` 和 `git config user.name/email` 是你要用的身份。

## 里程碑

每个里程碑过关要满足两条：①能白板讲清楚机制，②跑出可检验的结果。

### M0 环境、数据、集群摸底（10/21–10/28，约 10 GPU·h）
1. 搭仓库骨架、`git init`、写 .gitignore。本机用 uv 建 Python 3.12 环境（torch 2.5、e3nn==0.4.4、cuequivariance-torch、ase、matscipy、h5py、wandb、pytest）。mace-torch v0.3.14 装进 dev 依赖，只作对拍用。NSCC 在登录节点建同版本环境。
2. 下载 Zenodo 全部文件到 `local/zenodo/`，先看 sid 格式，判断 OMat24 要下哪几个子集。
3. NSCC 登录节点下载 OMat24 train 和修正版 val。直接读 aselmdb 按 sid 抽取。
   - 断言 795,721/795,721 个 ID 全部匹配；val 剔除稀土后断言为 657,756 帧。
4. 预处理成 HDF5 分片：positions、cell、Z、E、F、stress、sid、edge_index、整数 unit_shifts。抽完删掉原始数据。
5. **数据握手**：用作者 EnvTok 模型过一遍 HDF5，一次校验三件事：
   - 平均邻居数应为 40.5113；
   - 在 val 上算出的 MAE 量级要合理；
   - 应力符号与 ASE 约定一致。
6. 集群摸底：
   - `qstat -Q` 核对队列；
   - 确认 mpiexec 是哪种实现（Cray PALS 还是 OpenMPI）及其 rank 环境变量；
   - 确认有没有 aws-ofi-nccl；
   - 确认 SU 是否按 CPU/内存计费。
7. 测速：官方 `mace_run_train` 在 1×A100 上各跑 300 步，cueq 开一次、关一次。

### M1 模型从零实现与对拍（10/26–11/09）
1. 按顺序实现：
   - radial.py：Bessel 基在 Agnesi 变换后的 r̃ 上求值；多项式截断 p=5，作用在原始 r 上。
   - zbl.py：ZBL 排斥项。
   - 节点嵌入。
   - Density interaction：第 1 层用 skip_tp，第 2 层带残差。
   - symmetric contraction：U 矩阵与 mace 的 buffer 逐元素断言相等。
   - 读出：线性头和 16x0e 的 MLP。
   - 能量：E = ΣE0 + 读出 + ZBL。
   - 力与应力：用 `autograd.grad(E, [pos, ε], create_graph=training)` 同时求力和应力，应力用对称应变。
2. 对拍：
   - (a) 同配置随机初始化、权重映射后，fp64 下 E/F/S 误差 < 1e-10；
   - (b) 加载作者 EnvTok 模型，在 1,000 帧 val 上 fp32 误差 < 1e-5。
   - 结果写进 `docs/PARITY.md`。
3. 物理测试：旋转、平移、置换下保持等变/不变；有限差分验证力；有限应变验证应力。

### M2 训练栈 + 分布式逻辑（11/05–11/16，本机 CPU 用 gloo 后端测试）
实现“8 卡训练策略”第 4–8 节的全部逻辑：
- 均衡 sampler 和不补齐的 val sampler；
- 全局归一化、梯度累积（`no_sync`）；
- 早停广播、原子写 checkpoint、walltime 保护、NaN 保护；
- tracking 和 status.sh。

另外实现 UniversalLoss（与 mace 对拍）、参数分组（按参数名单测）、AMSGrad、EMA（带 warmup）、ReduceLROnPlateau，并写好 `eval/` 骨架和 ASE calculator。本机用 gloo 后端跑 1/2/4 进程，验证梯度等价和逐位续训。

### M3 NSCC 8 卡试跑（11/17–11/23，约 80 GPU·h）
1. cuEquivariance 后端接入，并通过 e3nn 与 cueq 的对拍。
2. 启动器：主方案和备选方案各跑一次 2 节点作业。NCCL 日志确认网卡和传输方式，测 all-reduce 带宽。
3. 一个 8 卡作业内跑完“策略”第 9、10 节的测量：
   - 1/2/4/8 卡扩展效率；
   - 均衡 sampler 开与关的对比；
   - 后端速度比；
   - 梯度等价和曲线等价。
4. 50k 子集上与官方 `mace_run_train` 做 8 卡 A/B。
5. 用实测数字更新 docs/PLAN.md 的预算和墙钟。
6. 应急：跨节点不稳时先按第 9 节顺序排障。实在不行，AI 分区的单节点 8 卡（aiq 队列）是备选，前提是这个账号能用。仍不行就退到单节点 4 卡 × 累积 2，数学上等价，但会如实写明。

### M4 正式训练（11/24–12/03）
- 按“8 卡训练策略”执行：glong 120 h 加预挂的接续作业，早停 patience 40，不设人为上限。
- 唯一的硬约束是账号总额度，给 M5 评测留约 120 GPU·h。
- 你每天跑一次 status.sh 把输出贴给我。每次接续后检查：val 曲线连续，EMA 和早停状态正确恢复。

### M5 基准评测（评测脚本在 M4 期间搭好、先用作者模型验证；自写模型 12/03–12/10，约 120 GPU·h，估算）
1. **先验证评测脚本**：用作者 EnvTok 模型跑我们的每个基准，复现他们已发布的数字。MatPES 必须复现；其余三项至少复现一部分样本。
2. 自写模型跑四个基准：
   - **MatPES PBE 2025.2**：只比力。报 Table 1 分箱的中位误差（bulk / non-bulk）和 bulk 力 MAE；分箱用作者发布的 OMAT-0 逐原子误差，按结构 ID 和原子序号对齐。
   - **MatCalc-Bench**：生成能、体积模量、剪切模量、热容。
   - **MD 稳定性**（MLIP-Arena，80 个结构，10 ps NPT，300→3000 K，0→500 GPa）：完成率和物理发散率。
   - **nebDFT2k**：1,656 条 Li⁺ 迁移路径，分有/无 Hubbard-U 两组，报 MAE / RMSE / 中位误差。
   - 基准推理是相互独立的任务，按结构拆成 8 份并行跑，不需要 DDP。
3. 对比与种子方差：横向比的对象是作者已发布的 EnvTok、DIRECT、rdm1–4 和 MACE-OMAT-0 结果。rdm1–4 的四种子离散度作为噪声底。
4. **ID / OOD 划分**：结果汇总成 `results/tables/id_ood_summary.csv`，每一行标明它属于哪一类：
   - **ID**：OMat24 val，与训练集同一生成协议；
   - **数据集迁移（近 OOD）**：MatPES bulk（另一套数据、DIRECT 采样）、MatCalc 近平衡性质；
   - **OOD**：MatPES non-bulk（论文原文称其在 OMat24 训练分布之外）、极端温压下的 MD 稳定性、NEB 过渡态。
5. **潜空间距离分析**（`analysis/latent.py`，约 15 GPU·h）：
   - 特征：自写模型两层 interaction 的不变（标量）节点特征拼起来，共 256 维，用 PCA 降到 50 维。做法同论文 S5，但论文用的是 MACE-OMAT-0 的特征。
   - 参考集：训练集中随机抽约 100 万个原子，PCA 在这上面拟合。
   - 测试集：OMat24 val 抽样、MatPES bulk、MatPES non-bulk。特征在 MatPES 力评测的同一次前向里顺带提取，不额外跑推理。
   - 度量：每个测试原子到参考集最近 k 个原子（k=5）的平均距离。用 torch 分块 cdist 在 GPU 上算，不引入 faiss。
   - 产出：
     - (a) 各测试集的距离分布图，看 ID 与 OOD 是否明显分开；
     - (b) 力误差随距离分位数变化的曲线，按测试集分开画；
     - (c) 距离与力误差之间的 Spearman 相关系数；
     - (d) 用作者 EnvTok 模型的特征重复一遍，与论文的潜空间结论（98.42% 的 EnvTok 描述符连成一片）对照。
6. 汇总到 `results/tables/`、`results/figures/`（含 `latent_distance_hist.png`、`error_vs_distance.png`）和 `docs/RESULTS.md`。工程指标包括：对拍误差、1→8 卡扩展效率、相对 mace-torch 的吞吐、总 GPU·h 与墙钟。

### M6 发布（12/10–12/14）
README 定稿：结果表、学习曲线、扩展效率图、复现步骤、引用（作者论文与仓库、MACE、OMat24、MatPES、各基准）。走一遍推送前检查，打 `v1.0`，问你确认后推送。模型权重放 GitHub Release。

### M7 新方法（待定）
主结果出来后，根据弱点选方向（例如 non-bulk 误差、MD 稳定性），再单独规划，预算用剩余额度。

## GPU 预算（1,562.5 A100·h）

| 用途 | GPU·h |
|---|---|
| M0 测速 + 杂项 | ~10 |
| M3 8 卡试跑（启动器、扩展效率、等价性、A/B） | ~80 |
| M4 正式训练（8 卡，不设上限，估算） | ~560–800（e3nn 回退则更多） |
| M5 基准评测 + ID/OOD 与潜空间分析（估算） | ~135 |
| 剩余（M7 或意外重跑） | ~535–775 |

## 验证

- 本机：`pytest tests/` 覆盖以下测试：
  - `test_equivariance`、`test_forces_fd`、`test_stress_strain`；
  - `test_parity_mace`：随机初始化对拍，加上作者 EnvTok 模型对拍；
  - `test_backend_parity`：e3nn 与 cueq 输出一致；
  - `test_loss_vs_mace`、`test_param_groups`、`test_neighbors`；
  - `test_balanced_sampler`：全局 batch 成员不变、各 rank 边数均衡、各 rank 之间不重叠；
  - `test_ddp_grad_equiv`：gloo 后端下，1 进程累积 4 步、2 进程累积 2 步、4 进程的梯度一致；
  - `test_early_stop_broadcast`、`test_walltime_guard`；
  - `test_latent_knn`：在玩具数据上，分块 cdist 的 kNN 距离与暴力解一致；PCA 与 sklearn 结果一致；
  - `test_resume_bitwise`：在 CPU 上，连续跑 N 步与“跑 N/2 步 + 续训 N/2 步”比较，参数逐位相同。
- NSCC：M3 的 8 卡作业产出“策略”第 10 节的五项证据，以及扩展效率表；M4 每次接续后检查曲线连续。
- 评测：先用作者模型复现他们已发布的基准数字，再报自写模型。

## 简历 bullet（草稿，数字等结果出来再填，每条都要能对应到 results/ 里的文件）

- Re-implemented the E(3)-equivariant MACE interatomic potential in PyTorch from scratch (matched mace-torch to <1e-10 in fp64) and trained it from random initialisation on ~6.5M atoms from OMat24, using multi-node PyTorch DDP on 8× NVIDIA A100 (2 nodes) with gradient accumulation, checkpoint/resume and experiment tracking; [Z]% scaling efficiency from 1 to 8 GPUs.
- Reproduced the energy/force/stress training setup of Moerman et al. (2026) and evaluated transferability on MatPES, MatCalc, MD-stability and Li-ion migration benchmarks: force MAE [X] meV/Å at [Y] GPU-hours, compared against the authors' released EnvTok, DIRECT and four random-seed models.
- Separated in-distribution from out-of-distribution performance (force MAE [A] vs [B] meV/Å) and tied force error to latent-space distance from the training set (Spearman ρ = [C]). （这条只在 M5 结果确实显示出明确关系时才写。）
- （第三条等 M7 做完再写。）

写法约束：
- “from scratch” 和 “matched mace-torch” 只在 M1 对拍通过后才写。
- “8× A100 (2 nodes)” 只在主训练确实用 2 节点跑完后才写。
- “controlled comparisons” 和自己的 “variation across seeds” 不能写，因为这两项用的是作者公开的模型。

## 批准后的收尾

- 本计划去掉账号信息后，落盘为仓库里的 `docs/PLAN.md`。
- 更新记忆 `project_envtok_mace.md`：
  - 旧脚手架已不存在，项目于 2026-10-21 从零重启；
  - 新决策：8 卡（2 节点 × 4）、不设上限、对比用作者模型、加入 ID/OOD 划分与潜空间分析、新方法待定；
  - 文件夹约定：local/ 不入库，最终推送 GitHub。
- 记忆里不写作者标注。
