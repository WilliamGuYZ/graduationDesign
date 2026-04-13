## ati-codegen — 基于大语言模型的算法代码生成（毕业设计）

对 **Qwen2.5-Coder-7B-Instruct** 基座模型进行 **LoRA 微调**，提升其算法代码生成能力，并使用 **pass@k** 指标进行定量评测。

### 技术要点

- **基座模型**：Qwen2.5-Coder-7B-Instruct
- **微调方法**：LoRA（PEFT），r=32, alpha=64，目标模块 q/k/v/o/gate/up/down_proj
- **数据集**：KodCode（10,000 条经代码验证的编程题，含 question/solution/test/test_info）
- **推理引擎**：vLLM（高效批量推理）
- **评测指标**：pass@k（k=1, 5, 10），通过子进程执行生成代码 + 测试用例验证

### 目录结构

```text
ati-codegen/
├── data/
│   ├── raw/                           # 原始数据
│   │   ├── KodCode.parquet            #   KodCode 原始 Parquet
│   │   ├── KodCode.jsonl              #   转换并验证后的 JSONL
│   │   └── mbpp.jsonl                 #   MBPP 数据集（备用）
│   └── processed/                     # 处理后数据
│       ├── KodCode_train.jsonl        #   训练集（90%）
│       └── KodCode_eval.jsonl         #   评测集（10%）
├── scripts/                           # 数据处理与工具脚本
│   ├── convet_KodCode_to_jsonl.py     #   Parquet → JSONL + 代码验证
│   ├── split_train_eval.py            #   训练/评测集划分
│   ├── check_max_seq_length.py        #   Token 长度统计
│   ├── validate_code_with_test.py     #   代码正确性验证
│   └── train_eval.sh                  #   一键全流程脚本
├── train/                             # 训练相关
│   ├── train.py                       #   LoRA 微调主脚本
│   ├── latest_lora_adapter.txt        #   最新适配器路径指针
│   └── outputs/                       #   训练输出（.gitignore 排除）
├── evaluation/                        # 评测相关
│   ├── eval_base_passk.py             #   基座模型 pass@k 评测
│   ├── eval_lora_passk.py             #   LoRA 模型 pass@k 评测
│   └── results/                       #   评测结果
│       ├── eval_base_passk.txt
│       └── eval_lora_passk.txt
├── models/                            # 本地模型权重（.gitignore 排除）
│   └── Qwen2.5-Coder-7B-Instruct/
├── requirements.txt
└── README.md
```

---

### 运行环境

本项目需要 **Linux + NVIDIA GPU** 环境。推荐使用 **云 GPU 平台**（如 AutoDL、矩池云、恒源云等）或自有 GPU 服务器。

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（云平台一般默认 Ubuntu） |
| GPU | NVIDIA，≥16GB 显存（推荐 24GB+，如 A100 / RTX 4090 / RTX 3090） |
| CUDA | 11.8 或 12.x（云平台镜像通常自带） |
| Python | 3.10+（云平台镜像通常自带） |
| 磁盘 | ≥50GB 可用（模型 14GB + 数据 + 训练输出） |

> **云 GPU 平台选卡建议**：
> - **预算充足**：A100-80G / A100-40G（训练 + 评测全程无压力）
> - **性价比首选**：RTX 4090-24G / RTX 3090-24G（完全够用）
> - **最低门槛**：RTX 3080-20G / V100-16G（需调低 batch size，详见「显存调优」章节）

---

### 云 GPU 环境配置（完整教程）

以下教程适用于 **AutoDL / 矩池云 / 恒源云 / 阿里云 / 腾讯云** 等主流平台。各平台大同小异，核心步骤一致。

#### 第一步：创建云 GPU 实例

1. 在云平台注册账号、充值
2. 选择 GPU 机型（参考上方选卡建议）
3. **选择镜像**（关键）：
   - 优先选带 **PyTorch + CUDA** 的预装镜像（如 `PyTorch 2.1 + CUDA 12.1`、`PyTorch 2.3 + CUDA 12.4` 等）
   - 这样可以跳过手动安装 PyTorch 和 CUDA 的步骤
4. 创建实例，等待启动

> **各平台连接方式**：
> - **AutoDL**：网页终端 或 `ssh -p <端口> root@connect.xxx.autolab.com`
> - **矩池云**：网页终端 或 SSH
> - **恒源云**：`ssh -p <端口> root@<IP>`
> - **阿里云/腾讯云**：标准 SSH `ssh root@<公网IP>`
>
> 具体命令在各平台「实例详情」页面都有显示，直接复制即可。

#### 第二步：连接到云服务器

```bash
# SSH 连接（替换为你的实际信息）
ssh -p <端口> root@<服务器地址>

# 连接后先确认 GPU 可用
nvidia-smi
```

确认输出中有 GPU 信息（型号、显存、CUDA 版本）。云平台的预装镜像通常已配好驱动，这一步应该直接通过。

#### 第三步：上传项目代码

```bash
# 方式一：git clone（推荐，如果项目已推送到 GitHub / Gitee）
git clone <你的仓库地址> graduationDesign
cd graduationDesign/ati-codegen

# 方式二：从本地电脑上传（在你的本地电脑执行）
# scp -P <端口> -r ati-codegen/ root@<服务器地址>:/root/
# 然后在云服务器上：cd /root/ati-codegen

# 方式三：各平台的网盘/数据盘功能
# AutoDL：可以通过「文件存储」上传整个文件夹
# 矩池云 / 恒源云：类似，参考各平台文档
```

#### 第四步：安装依赖

```bash
# 如果平台提供了 conda 环境，先激活（可选）
# conda activate base

# 确认 Python 和 PyTorch 是否已预装
python3 --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

**如果上述命令输出正常**（PyTorch 已安装、CUDA 可用），直接安装其余依赖：

```bash
# 使用清华源加速（国内云服务器强烈推荐）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**如果云平台镜像没有预装 PyTorch**，需要先手动安装：

```bash
# 查看 CUDA 版本
nvidia-smi   # 右上角显示 CUDA Version

# 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 12.x：
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8：
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# 然后安装其余依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **验证安装**：
> ```bash
> python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
> python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
> ```
> 两条命令都不报错即可。

#### 第五步：下载基座模型

将 Qwen2.5-Coder-7B-Instruct 模型下载到 `models/` 目录：

```bash
# 方式一：modelscope（国内云服务器推荐，速度快）
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir models/Qwen2.5-Coder-7B-Instruct

# 方式二：huggingface-cli（需要能访问 huggingface.co）
# pip install huggingface_hub
# huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir models/Qwen2.5-Coder-7B-Instruct
```

> 下载约 14GB。国内云服务器用 modelscope 通常几分钟就能下完。
> 下载完成后确认 `models/Qwen2.5-Coder-7B-Instruct/` 下有 `config.json`、`*.safetensors` 等文件。
>
> **磁盘空间提示**：如果系统盘空间不够，可以把模型下载到数据盘，再做软链接：
> ```bash
> # 以 AutoDL 为例，数据盘在 /root/autodl-tmp/
> modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir /root/autodl-tmp/Qwen2.5-Coder-7B-Instruct
> ln -s /root/autodl-tmp/Qwen2.5-Coder-7B-Instruct models/Qwen2.5-Coder-7B-Instruct
> ```

#### 第六步：数据准备

> **如果仓库中已有 `data/raw/KodCode.jsonl` 和 `data/processed/KodCode_train.jsonl`、`KodCode_eval.jsonl`，可以跳过本步骤，直接进入第七步。**

```bash
# 1. 如果只有 Parquet，先转换为 JSONL（会执行代码验证，耗时较长，约 30-60 分钟）
#    如果 data/raw/KodCode.jsonl 已存在则跳过此步
python3 scripts/convet_KodCode_to_jsonl.py

# 2. 划分训练/评测集（9:1）
#    如果 data/processed/ 下已有 KodCode_train.jsonl 和 KodCode_eval.jsonl 则跳过此步
python3 scripts/split_train_eval.py --ratio 0.9 --no-shuffle
```

#### 第七步：运行全流程

> **重要：云 GPU 上运行长任务务必使用 `tmux` 或 `nohup`，防止 SSH 断连导致训练中断！**

```bash
# ===== 推荐：使用 tmux（可随时断开重连查看进度）=====
tmux new -s train
bash scripts/train_eval.sh
# 断开 SSH 不影响运行：按 Ctrl+B 然后按 D
# 重新连接后查看：tmux attach -t train

# ===== 或者：使用 nohup（后台运行，日志写入文件）=====
# nohup bash scripts/train_eval.sh > run.log 2>&1 &
# tail -f run.log   # 查看实时日志
```

`train_eval.sh` 依次执行：数据集划分 → LoRA 训练 → 基座评测 → LoRA 评测。

**也可以单步运行**：

```bash
# 1. LoRA 微调（约 1-3 小时，取决于 GPU 型号）
python3 train/train.py

# 2. 基座模型评测（约 20-40 分钟）
python3 evaluation/eval_base_passk.py

# 3. LoRA 模型评测（约 20-40 分钟，自动读取最新 adapter）
python3 evaluation/eval_lora_passk.py
```

#### 第八步：查看结果

```bash
# 评测结果
cat evaluation/results/eval_base_passk.txt
cat evaluation/results/eval_lora_passk.txt

# TensorBoard 训练曲线（可选）
tensorboard --logdir train/outputs/ --bind_all
# 浏览器访问 http://<服务器IP>:6006
# AutoDL 用户：在实例详情页点「自定义服务」可直接访问 6006 端口
```

#### 第九步：下载结果到本地

训练和评测完成后，把结果下载到本地电脑：

```bash
# === 以下命令在你的本地电脑执行（替换为你的实际信息）===

# 下载评测结果
scp -P <端口> root@<服务器地址>:/root/graduationDesign/ati-codegen/evaluation/results/*.txt ./

# 下载训练好的 LoRA adapter（如需后续部署或展示）
scp -P <端口> -r root@<服务器地址>:/root/graduationDesign/ati-codegen/train/outputs/ ./train_outputs/

# 下载 TensorBoard 日志（可选，用于本地查看训练曲线）
# scp -P <端口> -r root@<服务器地址>:/root/graduationDesign/ati-codegen/train/outputs/*/runs/ ./tb_logs/
```

> **省钱提醒**：训练和评测完成后记得**关机**！云 GPU 按时计费，不用时务必停止实例。

---

### 显存调优

如果遇到 OOM（显存不足），可以调整以下参数：

**训练** (`train/train.py` 顶部)：

| 参数 | 默认值 | 显存不足时 |
|------|--------|-----------|
| `BATCH_SIZE` | 4 | 降为 2 或 1 |
| `GRADIENT_ACCUMULATION` | 8 | 相应增大以保持等效 batch |
| `MAX_LENGTH` | 1024 | 降为 512 |

**评测** (`evaluation/eval_base_passk.py` 和 `eval_lora_passk.py` 顶部)：

| 参数 | 默认值 | 显存不足时 |
|------|--------|-----------|
| `GPU_MEMORY_UTILIZATION` | 0.9 | 降为 0.8 或 0.7 |
| `VLLM_MAX_MODEL_LEN` | 2048 | 降为 1024 |
| `NUM_SAMPLES` | 10 | 降为 5（但会影响 pass@10） |

> **各显卡参考配置**：
> - **A100-80G / A100-40G**：全部默认值即可
> - **RTX 4090-24G / RTX 3090-24G**：默认值即可，评测时 `GPU_MEMORY_UTILIZATION` 可降为 0.85
> - **V100-16G / RTX 3080-20G**：训练 `BATCH_SIZE=1, GRADIENT_ACCUMULATION=32, MAX_LENGTH=512`；评测 `GPU_MEMORY_UTILIZATION=0.7, VLLM_MAX_MODEL_LEN=1024`

---

### 数据格式

每条训练/评测数据为一行 JSON，格式如下：

```json
{
  "question": "题目描述...",
  "solution": "def func(x):\n    ...",
  "test": "def test_func():\n    assert func(1) == 2\n    ...",
  "test_info": [
    {
      "function_declaration": "def func(x):",
      "function_name": "func",
      "parameter_list": "x"
    }
  ]
}
```

### 评测结果

在 KodCode 评测集（1000 题）上的 pass@k 结果：

| 指标 | 基座模型 | LoRA 微调后 | 提升 |
|------|---------|------------|------|
| **pass@1** | 59.60% | 67.50% | +7.90% |
| **pass@5** | 73.00% | 84.00% | +11.00% |
| **pass@10** | 77.70% | 88.70% | +11.00% |

> 参数：每题采样 10 次，temperature=0.7，子进程执行超时 5 秒。

---

### 常见问题

**Q: `nvidia-smi` 找不到命令？**
A: NVIDIA 驱动未安装。云平台上一般不会出现——检查是否选错了 CPU 实例。如果是自有服务器，需要手动安装驱动：`sudo apt install -y nvidia-driver-535 && sudo reboot`。

**Q: `torch.cuda.is_available()` 返回 `False`？**
A: PyTorch 的 CUDA 版本与系统不匹配。重新安装：`pip install torch --index-url https://download.pytorch.org/whl/cu121`。云平台上如果选了预装 PyTorch 镜像，一般不会出现此问题。

**Q: `pip install vllm` 特别慢或失败？**
A: 国内云服务器用清华源：`pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple`。如果编译报错，确认 CUDA 版本 ≥ 11.8 且有 gcc（`sudo apt install -y build-essential`）。

**Q: SSH 断开后训练就停了？**
A: 长任务必须用 `tmux` 或 `nohup` 运行。参考第七步中的说明。

**Q: 训练时 OOM？**
A: 降低 `train/train.py` 中的 `BATCH_SIZE`（如从 4 降为 2），同时增大 `GRADIENT_ACCUMULATION`（如从 8 增到 16）。参考「显存调优」章节中各显卡的参考配置。

**Q: 评测时 OOM？**
A: 降低评测脚本中的 `GPU_MEMORY_UTILIZATION`（如从 0.9 降为 0.7）和 `VLLM_MAX_MODEL_LEN`（如从 2048 降为 1024）。

**Q: `latest_lora_adapter.txt` 路径不对？**
A: 该文件在训练完成后自动生成，内容为 adapter 的绝对路径。如果把项目移到了新位置，需要手动修改文件内容指向正确路径，或重新训练。

**Q: 云平台的数据盘在哪？**
A: 各平台不同。AutoDL 一般在 `/root/autodl-tmp/`（数据盘）和 `/root/autodl-fs/`（文件存储）。建议把模型下载到数据盘以节省系统盘空间，然后做软链接（参考第五步）。

---

### 注意事项

1. **操作系统**：必须在 Linux 上运行（vLLM 不支持 Windows / macOS）
2. **模型路径硬编码**：`models/Qwen2.5-Coder-7B-Instruct` 在多个脚本中硬编码，请确保模型放在此路径下（或通过软链接指向实际位置）
3. **`latest_lora_adapter.txt`**：训练完成后自动写入适配器路径，LoRA 评测脚本读取此文件
4. **数据验证耗时**：`convet_KodCode_to_jsonl.py` 会 exec 每条代码做正确性验证，10,000 条可能需要较长时间
5. **磁盘空间**：模型约 14GB + 训练输出约 1-2GB，确保至少 50GB 可用空间
6. **省钱**：云 GPU 按时计费，完成后务必关机；长时间不用建议把结果下载到本地后释放实例
