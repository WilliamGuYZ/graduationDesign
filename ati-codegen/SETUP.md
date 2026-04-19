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

本项目需要 **Linux + NVIDIA GPU** 环境：在 **远程 Ubuntu GPU 服务器**（实验室机器或租用 GPU）上完成训练与 vLLM 评测；**本地 Windows** 通过 **WSL2 或 Linux 虚拟机** 使用 SSH 连接服务器、管理 Git 与拷贝结果。


| 项目      | 要求                                              |
| ------- | ----------------------------------------------- |
| 服务器 OS | Linux（推荐 Ubuntu 22.04 / 20.04）                   |
| GPU     | NVIDIA，≥16GB 显存（推荐 24GB+）                       |
| CUDA    | 11.8 或 12.x                                     |
| Python  | 3.10+（本仓库 `requirements.txt` 说明中推荐 3.11）      |
| 磁盘      | 系统盘 ≥40GB；另建议 ≥100GB 可用空间（模型约 14GB + 数据 + 训练输出） |
| 本地电脑    | Windows 10/11 + WSL2 或 Linux 虚拟机（SSH 客户端）        |

---

### 为什么需要 Linux + NVIDIA GPU

本项目涉及大语言模型（7B 参数）的 LoRA 微调和 vLLM 批量推理，这两个核心环节对操作系统和硬件有硬性要求：

#### 为什么必须用 Linux


| 组件                 | 为什么依赖 Linux        | 说明                                                                                                            |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------- |
| **vLLM**           | 仅支持 Linux          | vLLM 是本项目的推理引擎，用于高效批量生成代码。它依赖 Linux 内核特性（如 CUDA IPC、共享内存管理），**官方明确不支持 Windows 和 macOS**。没有 vLLM，pass@k 评测无法执行 |
| **PyTorch + CUDA** | Linux 支持最完善        | PyTorch 的 GPU 训练在 Linux 上性能最优且兼容性最好，NVIDIA 官方驱动和 CUDA 工具链优先保证 Linux 平台                                        |
| **PEFT / LoRA 微调** | 依赖 PyTorch CUDA 后端 | LoRA 训练需要将模型参数和梯度放在 GPU 显存中，底层依赖 CUDA 内核算子，这些在 Linux 上经过充分测试和优化                                               |
| **子进程代码执行**        | Linux 进程模型更稳定      | pass@k 评测需要在沙箱子进程中执行上千段生成的代码，Linux 的 fork/exec 机制和信号处理比 Windows 更可靠                                           |


> **简单理解**：vLLM（推理引擎）只能在 Linux 上运行，这是最根本的限制。即使其他组件理论上可以在 Windows 运行，vLLM 的 Linux-only 限制决定了整个项目必须部署在 Linux 环境中。

#### 为什么必须用 NVIDIA GPU


| 环节                | GPU 需求  | 说明                                                                                                                               |
| ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **LoRA 微调**       | 必须      | 7B 参数模型即使使用 LoRA（仅训练约 1% 参数），模型本体仍需全部加载到显存中做前向/反向传播。FP16 下模型约占 14GB 显存，加上梯度和优化器状态，至少需要 16GB+ 显存。纯 CPU 训练理论可行但耗时将从小时级变为天/周级，完全不实际 |
| **vLLM 推理**       | 必须      | vLLM 使用 PagedAttention 等 GPU 优化技术实现高吞吐推理。pass@k 评测需要对 1000 道题各采样 10 次（共 10,000 次推理），GPU 推理约 20-40 分钟完成，CPU 推理可能需要数天              |
| **为什么必须是 NVIDIA** | CUDA 生态 | PyTorch、vLLM、Flash Attention 等核心依赖均基于 NVIDIA CUDA 构建。AMD ROCm 和 Intel oneAPI 的支持尚不成熟，vLLM 对非 NVIDIA GPU 的支持非常有限                  |


> **显存需求总结**：
>
> - **最低 16GB 显存**（如 V100-16G）：可以运行，但需调低 batch size 和序列长度
> - **推荐 24GB+ 显存**（如 A10-24G）：默认参数即可顺畅运行
> - **消费级显卡不推荐**：RTX 3060-12G 等显存不足 16GB 的卡无法加载完整的 7B 模型

#### 本地电脑的角色

本项目中，你的 **Windows 本地电脑不直接运行训练/评测代码**，它的角色是：

1. **编写和管理代码**（在 IDE 中开发、用 Git 管理版本）
2. **通过 SSH 连接到远程 GPU 服务器**（在本地 WSL 或虚拟机的 Linux 终端中操作）
3. **与服务器同步代码**（`git clone` / `git pull`，必要时 `scp`）
4. **在服务器上执行训练和评测**（计算在远端 GPU 上完成）
5. **从服务器取回结果到本地**（评测结果、训练日志等）

```text
┌──────────────────────┐         SSH          ┌────────────────────────────┐
│  Windows 本地电脑     │ ──────────────────→  │  远程 Ubuntu GPU 服务器     │
│                      │                      │                            │
│  ┌────────────────┐  │   同步代码 (git/scp)  │  Ubuntu + NVIDIA GPU       │
│  │ WSL2 / 虚拟机   │  │ ──────────────────→  │  PyTorch + vLLM + PEFT     │
│  │ (SSH 客户端)    │  │                      │                            │
│  └────────────────┘  │   取回结果 (scp)      │  训练 → 评测 → 结果输出     │
│                      │ ←──────────────────   │                            │
│  IDE / Git           │                      │                            │
└──────────────────────┘                      └────────────────────────────┘
```

---

### 本地 Linux 环境搭建（WSL / 虚拟机）

由于 SSH、scp、git 等命令行工具在 Linux 终端中使用最为便捷，建议在 Windows 电脑上搭建一个本地 Linux 环境。以下提供两种方案：

#### 方案一：WSL2（推荐，轻量快速）

WSL（Windows Subsystem for Linux）是 Windows 内置的 Linux 子系统，无需安装完整虚拟机，启动快、资源占用低。

```powershell
# ===== 在 Windows PowerShell（管理员）中执行 =====

# 1. 安装 WSL2（Windows 10 2004+ 或 Windows 11）
wsl --install

# 安装完成后会提示重启电脑，重启后自动弹出 Ubuntu 终端
# 设置 Linux 用户名和密码（这是 WSL 内部的账号，与 Windows 无关）

# 2. 如果已安装过 WSL，确认版本为 WSL2
wsl --set-default-version 2

# 3. 如果想安装指定版本的 Ubuntu
wsl --install -d Ubuntu-22.04
```

安装完成后，可以通过以下方式打开 WSL 终端：

- 在 Windows 搜索栏输入 `Ubuntu` 或 `WSL`
- 在 Windows Terminal 中选择 Ubuntu 标签页
- 在 VS Code / Cursor 中打开终端，选择 WSL

```bash
# ===== 在 WSL 终端中执行（首次使用建议更新）=====

# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装常用工具
sudo apt install -y git openssh-client

# 验证 SSH 可用
ssh -V
# 输出类似：OpenSSH_8.9p1 Ubuntu-3ubuntu0.6, OpenSSL 3.0.2
```

> **WSL 访问 Windows 文件**：
> Windows 的 C 盘在 WSL 中挂载为 `/mnt/c/`，因此你的项目目录可以通过以下路径访问：
>
> ```bash
> # 假设项目在 Windows 桌面
> cd /mnt/c/Users/<你的用户名>/Desktop/graduationDesign/ati-codegen
> # 或者 D 盘
> cd /mnt/d/My\ Documents/<路径>/graduationDesign/ati-codegen
> ```

#### 方案二：虚拟机（VMware / VirtualBox）

如果 WSL 安装遇到问题（如公司电脑限制 Hyper-V），可以使用虚拟机：

1. 下载 [VMware Workstation Player](https://www.vmware.com/products/workstation-player.html)（免费）或 [VirtualBox](https://www.virtualbox.org/)（免费开源）
2. 下载 [Ubuntu 22.04 ISO 镜像](https://ubuntu.com/download/desktop)
3. 创建虚拟机：分配 2 核 CPU、4GB 内存、40GB 磁盘即可（仅用作 SSH 客户端，无需高配）
4. 安装 Ubuntu，完成后：

```bash
# 安装必要工具
sudo apt update && sudo apt install -y git openssh-client

# 验证
ssh -V
git --version
```

> **虚拟机与主机共享文件**：
>
> - **VMware**：设置「共享文件夹」，在虚拟机中通过 `/mnt/hgfs/` 访问
> - **VirtualBox**：安装增强功能后，设置共享文件夹，挂载到自定义路径

#### 方案对比


|              | WSL2         | 虚拟机                   |
| ------------ | ------------ | --------------------- |
| 安装难度         | 一条命令         | 需下载 ISO + 安装系统        |
| 启动速度         | 秒级           | 分钟级                   |
| 资源占用         | 极低           | 需分配独立 CPU/内存          |
| 文件共享         | 自动（/mnt/c/）  | 需手动配置共享文件夹            |
| 与 Windows 集成 | 无缝（共享网络、剪贴板） | 需额外配置                 |
| 适用场景         | SSH 客户端、日常开发 | SSH 客户端、需要完整 Linux 桌面 |


**推荐使用 WSL2**，操作最简单，且 Windows Terminal 和 Cursor/VS Code 都能直接集成。

---

### 团队共享 Ubuntu 服务器（WSL / 虚拟机 SSH + `~/yejunyin` + Conda 共享环境）

在实验室或课程提供的 **Ubuntu GPU 服务器** 上完成训练与评测时，可按下面流程操作。说明如何从 **Windows 上的 WSL2 或 Linux 虚拟机** 用终端 **SSH 登录**，在 **`~/yejunyin`** 下从 GitHub **克隆本仓库**，配置 **团队共用的 Conda 环境**，按 **`requirements.txt`** 安装依赖，以及 **Git 同步与成员协作**。

> **安全提示**：SSH 密码、密钥、面板截图等 **不要写入 Git**、不要发到公开群聊。密码由管理员或实验平台单独下发；连接信息变更时只更新本文档中的 **主机 / 端口 / 用户名**，不要写明文密码。

#### 1. 在 WSL / 虚拟机里安装 SSH 客户端

若尚未安装，在 **WSL Ubuntu** 或 **虚拟机 Ubuntu** 中执行：

```bash
sudo apt update && sudo apt install -y git git-lfs openssh-client
git lfs install
ssh -V
```

#### 2. SSH 连接服务器（非标准端口）

在 **本地 WSL / 虚拟机终端** 中执行（`-p` 指定远端 SSH 端口；用户名与 IP 以实验平台为准）：

```bash
ssh -p 8552 ubuntu@116.172.94.6
```

首次连接会提示确认主机指纹，输入 `yes`；随后按提示输入密码（输入时终端不显示字符，属正常现象）。

> **可选：免密登录**：在本地生成 SSH 密钥后，将公钥追加到服务器对应用户的 `~/.ssh/authorized_keys`（需至少一次能用密码登录以完成配置）。

#### 3. 在 `~/yejunyin` 下克隆仓库

**以下命令在已成功 SSH 登录的「服务器」上执行**（不是 WSL 里，除非你是在说明本地目录；此处指远端 `ubuntu@服务器`）：

```bash
mkdir -p ~/yejunyin
cd ~/yejunyin

# 若尚未配置 Git 用户信息（提交用，按需）
# git config --global user.name "你的名字"
# git config --global user.email "你的邮箱"

git clone https://github.com/WilliamGuYZ/graduationDesign.git graduationDesign
cd graduationDesign/ati-codegen

# 本仓库含 Git LFS 大文件，务必执行（否则 JSONL 可能是指针文件，脚本会报错）
git lfs pull
```

若团队使用 **私有仓库** 或 **Fork**，将上述 `git clone` 地址换成实际地址；本地目录名建议仍为 `graduationDesign`，与文档其余路径一致。

#### 4. 配置「共享」Conda 环境（推荐：`--prefix` 固定路径）

团队共用一台机器时，建议由 **一位管理员** 在 **`~/yejunyin` 下的固定路径** 创建环境，其他人只 **激活同一前缀**，避免每人一套环境占满磁盘。

**4.1 管理员首次创建（服务器上已安装 Miniconda/Anaconda 的前提下）**

```bash
mkdir -p ~/yejunyin/conda-envs

# 使用前缀路径创建环境（环境名路径可自定，全组统一即可）
conda create --prefix ~/yejunyin/conda-envs/qwen-coder python=3.11 -y

# 激活该环境（以后每次登录都要先激活再跑训练/评测）
conda activate ~/yejunyin/conda-envs/qwen-coder
```

若 `conda activate` 对前缀路径报错，先执行一次 `conda init bash` 并重新打开 shell，或按 Conda 官方说明启用 `conda shell` 集成。

**4.2（可选）共享下载缓存，节省磁盘**

```bash
mkdir -p ~/yejunyin/conda-pkgs
conda config --add pkgs_dirs ~/yejunyin/conda-pkgs
```

多人并行安装时仍可能产生锁竞争，尽量由管理员统一装好依赖，成员只做 `git pull` 与运行。

#### 5. 安装依赖（与 `requirements.txt` 头部说明一致）

激活共享环境后，在 **`ati-codegen` 目录** 下按 **`requirements.txt`** 中的顺序操作（**务必先配 pip 镜像、再装 PyTorch、再 `pip install -r`**）：

```bash
cd ~/yejunyin/graduationDesign/ati-codegen
conda activate ~/yejunyin/conda-envs/qwen-coder

# 1）pip 使用国内镜像（与 requirements.txt 注释一致）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 2）先安装 PyTorch（CUDA 版本需与服务器驱动匹配；以下为 requirements.txt 中的 cu121 示例）
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# 3）其余依赖一键安装
pip install -r requirements.txt
```

若服务器 CUDA / 驱动与 `cu121`  wheel 不匹配，请根据 `nvidia-smi` 与 PyTorch 官网说明改用 `cu118` 等对应命令，再执行 `pip install -r requirements.txt`。

安装完成后可在服务器上验证：

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
```

#### 6. Git 同步修改（日常开发）

**在服务器项目目录中**：

```bash
cd ~/yejunyin/graduationDesign/ati-codegen
git status
git pull
git lfs pull    # 仅当大文件/LFS 指针有更新时
```

若 `requirements.txt` 有变更，激活共享环境后补装：

```bash
conda activate ~/yejunyin/conda-envs/qwen-coder
pip install -r requirements.txt
```

本地 Windows / 其他电脑修改代码后：**先 `commit` + `push` 到 GitHub**，再到服务器 **`git pull`**（大文件变更时 **`git lfs pull`**），保持与仓库一致。

#### 7. 首次配置完成后，其他团队成员怎么用

1. 从管理员或平台获取 **SSH 主机、端口、账号、密码**（或密钥），在 **WSL / 虚拟机** 中执行 `ssh -p <端口> ubuntu@<主机>` 登录。  
2. `cd ~/yejunyin/graduationDesign/ati-codegen`（若仓库尚未克隆，则按 **第 3 节** 执行 `git clone` + `git lfs pull`）。  
3. `conda activate ~/yejunyin/conda-envs/qwen-coder`（路径与管理员约定一致）。  
4. `git pull`（及按需 `git lfs pull`）。  
5. 按 README 其余章节下载模型、运行 `train/`、`evaluation/` 或 `scripts/`（多人共用同一 `ubuntu` 账号时，建议用 **tmux** 或 **不同工作目录** 避免互相覆盖进程与输出）。

> **路径约定**：下文默认项目在 **`~/yejunyin/graduationDesign/ati-codegen`**；若你的克隆路径不同，请相应替换命令中的目录。训练/评测脚本以 **`ati-codegen` 内相对路径** 为准。

---

### 模型下载与训练评测流程（GPU 服务器）

在已完成 **SSH 登录、克隆仓库、`git lfs pull`、Conda 环境与 `requirements.txt` 安装** 后，在服务器项目目录中继续：

```bash
cd ~/yejunyin/graduationDesign/ati-codegen
```

#### 下载基座模型

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir models/Qwen2.5-Coder-7B-Instruct
```

下载约 14GB。也可使用 `huggingface-cli`（需能访问 Hugging Face）。完成后确认 `models/Qwen2.5-Coder-7B-Instruct/` 下有 `config.json`、`*.safetensors` 等。

> **路径对齐**：脚本默认读取 `models/Qwen2.5-Coder-7B-Instruct`。若模型放在其他挂载目录，请在 `ati-codegen/models/` 下用 **软链接** 指向真实路径，否则会报 `HFValidationError: Repo id must be in the form ...`。

#### 数据准备

> 若仓库中已有 `data/raw/KodCode.jsonl` 与 `data/processed/KodCode_train.jsonl`、`KodCode_eval.jsonl`，可跳过本节。

```bash
head -n 1 data/raw/KodCode.jsonl
# 若首行为 Git LFS 指针，在仓库根目录执行：git lfs pull

python3 scripts/convet_KodCode_to_jsonl.py   # 仅当需从 Parquet 生成 JSONL 时，耗时可较长
python3 scripts/split_train_eval.py --ratio 0.9 --no-shuffle
```

#### 运行训练与评测

> 长任务务必使用 **`tmux`** 或 **`nohup`**，避免 SSH 断连中断训练。

```bash
tmux new -s train
bash scripts/train_eval.sh
# Ctrl+B 后按 D 脱离会话；再次连接：tmux attach -t train
```

也可分步执行：

```bash
python3 train/train.py
python3 evaluation/eval_base_passk.py
python3 evaluation/eval_lora_passk.py
```

#### 查看结果

```bash
cat evaluation/results/eval_base_passk.txt
cat evaluation/results/eval_lora_passk.txt
# TensorBoard（可选）：tensorboard --logdir train/outputs/ --bind_all
# 从本机浏览器访问时，需通过 SSH 隧道或在服务器侧放行对应端口（如 6006）
```

#### 将结果复制到本机

在 **本地 WSL / 虚拟机** 终端（未 SSH 进服务器）中，示例（端口、用户、路径按实际修改）：

```bash
mkdir -p evaluation/results train/outputs
scp -P 8552 ubuntu@116.172.94.6:~/yejunyin/graduationDesign/ati-codegen/evaluation/results/*.txt ./evaluation/results/
scp -r -P 8552 ubuntu@116.172.94.6:~/yejunyin/graduationDesign/ati-codegen/train/outputs/ ./train/
```

---

### 显存调优

如果遇到 OOM（显存不足），可以调整以下参数：

**训练** (`train/train.py` 顶部)：


| 参数                      | 默认值  | 显存不足时           |
| ----------------------- | ---- | --------------- |
| `BATCH_SIZE`            | 4    | 降为 2 或 1        |
| `GRADIENT_ACCUMULATION` | 8    | 相应增大以保持等效 batch |
| `MAX_LENGTH`            | 1024 | 降为 512          |


**评测** (`evaluation/eval_base_passk.py` 和 `eval_lora_passk.py` 顶部)：


| 参数                       | 默认值  | 显存不足时              |
| ------------------------ | ---- | ------------------ |
| `GPU_MEMORY_UTILIZATION` | 0.9  | 降为 0.8 或 0.7       |
| `VLLM_MAX_MODEL_LEN`     | 2048 | 降为 1024            |
| `NUM_SAMPLES`            | 10   | 降为 5（但会影响 pass@10） |


> **按显存粗调的参考**：
>
> - **约 80GB 显存**：训练与评测可用默认参数
> - **约 24GB 显存**：默认一般可用；评测可将 `GPU_MEMORY_UTILIZATION` 降至 0.85
> - **16GB 显存**：训练建议 `BATCH_SIZE=1, GRADIENT_ACCUMULATION=32, MAX_LENGTH=512`；评测 `GPU_MEMORY_UTILIZATION=0.7, VLLM_MAX_MODEL_LEN=1024`

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


| 指标          | 基座模型   | LoRA 微调后 | 提升      |
| ----------- | ------ | -------- | ------- |
| **pass@1**  | 59.60% | 67.50%   | +7.90%  |
| **pass@5**  | 73.00% | 84.00%   | +11.00% |
| **pass@10** | 77.70% | 88.70%   | +11.00% |


> 参数：每题采样 10 次，temperature=0.7，子进程执行超时 5 秒。

---

### 常见问题

#### 本地环境相关

**Q: WSL 安装失败，提示需要启用虚拟化？**
A: 需要在 BIOS 中开启 Intel VT-x 或 AMD-V（虚拟化技术），然后在 Windows 功能中启用「适用于 Linux 的 Windows 子系统」和「虚拟机平台」。具体步骤因电脑品牌而异，搜索「你的电脑品牌 + 开启虚拟化」。如果确实无法开启 WSL，改用虚拟机方案。

**Q: WSL 中 `ssh` 命令找不到？**
A: 执行 `sudo apt update && sudo apt install -y openssh-client` 安装 SSH 客户端。

**Q: WSL 中怎么访问 Windows 上的项目文件？**
A: Windows 磁盘自动挂载在 `/mnt/` 下。C 盘 → `/mnt/c/`，D 盘 → `/mnt/d/`。路径中的空格需要用 `\` 转义或用引号括起来，如 `cd "/mnt/d/My Documents/"`。

**Q: 能不能不用 WSL / 虚拟机，直接在 Windows 的 CMD/PowerShell 中 SSH？**
A: 可以。Windows 10 1809+ 自带 OpenSSH 客户端，在 PowerShell 中执行 `ssh -p <端口> <用户>@<主机>` 即可。但 `scp` 等路径处理在 Linux 终端往往更省事，因此仍推荐 WSL。

#### 远程 GPU 服务器相关

**Q: `nvidia-smi` 找不到命令？**
A: 服务器未正确安装 NVIDIA 驱动，或当前无 GPU。在 Ubuntu 上可尝试：`sudo apt update && sudo apt install -y nvidia-driver-535`，然后 `sudo reboot`，再重新登录后执行 `nvidia-smi`。具体以机房/云厂商文档为准。

**Q: `torch.cuda.is_available()` 返回 `False`？**
A: 多为 PyTorch 与驱动/CUDA 不匹配，或装成了 CPU 版 wheel。请按 `nvidia-smi` 显示的驱动与官方说明重装带 CUDA 的 PyTorch（例如 `pip install torch --index-url https://download.pytorch.org/whl/cu121`），并与本文「安装依赖」及 `requirements.txt` 中的版本说明对齐。

**Q: `pip install torch` / `pip install sympy` 下载很慢或超时？**
A: 先升级 pip，并配置国内 PyPI 镜像（见上文「安装依赖」中的 `pip config set global.index-url`）。旧版 pip（如 22.x）可能仍把部分下载重定向到国外源。安装 torch 前可先 `pip install sympy networkx filelock`，再安装 PyTorch wheel。

**Q: `pip install vllm` 特别慢或失败？**
A: 使用国内镜像安装：`pip install vllm -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`。若编译失败，确认 CUDA 版本 ≥ 11.8 且已安装编译工具：`sudo apt install -y build-essential`。

**Q: `bash scripts/train_eval.sh` 报 `set: pipefail: invalid option name`？**
A: 脚本文件是 Windows 的 CRLF 换行导致的（实际变成 `pipefail\\r`）。在远程 Linux 服务器上执行：
```bash
apt update && apt install -y dos2unix
dos2unix scripts/*.sh
```

**Q: SSH 断开后训练就停了？**
A: 长任务必须用 `tmux` 或 `nohup` 运行，见上文「运行训练与评测」。断开 SSH 不会结束服务器上已脱离的 `tmux` 会话。

**Q: 训练时 OOM？**
A: 降低 `train/train.py` 中的 `BATCH_SIZE`（如从 4 降为 2），同时增大 `GRADIENT_ACCUMULATION`（如从 8 增到 16）。参考「显存调优」章节中按显存的参考配置。

**Q: 评测时 OOM？**
A: 降低评测脚本中的 `GPU_MEMORY_UTILIZATION`（如从 0.9 降为 0.7）和 `VLLM_MAX_MODEL_LEN`（如从 2048 降为 1024）。

**Q: `latest_lora_adapter.txt` 路径不对？**
A: 该文件在训练完成后自动生成，内容为 adapter 的绝对路径。如果把项目移到了新位置，需要手动修改文件内容指向正确路径，或重新训练。

**Q: 额外数据盘没有自动挂载？**
A: 云主机或实验室机器若挂了第二块盘，通常需要在系统里手动分区、`mkfs`、`mount` 并写入 `/etc/fstab`。挂载后可将大文件（模型、数据集）放在该盘上，并在项目 `models/` 下用软链接指向。

**Q: `data/raw/KodCode.jsonl` 第一行是 `version https://git-lfs.github.com/spec/v1`？**
A: 这是 Git LFS 的**指针文件**，说明你没有把 LFS 大文件拉下来（常见于未安装 git-lfs 或未执行 `git lfs pull`）。在**仓库根目录**执行：
```bash
apt update && apt install -y git-lfs
git lfs install
git lfs pull
```
拉取完成后 `head -n 1 data/raw/KodCode.jsonl` 应该能看到以 `{` 开头的 JSON 行。

**Q: `JSONDecodeError: Expecting value`（但不是 LFS 指针）？**
A: 通常是文件开头带 BOM、夹杂空行、或某一行被截断/混入日志。`scripts/split_train_eval.py` 已增强：会自动跳过空行、去 BOM，并在报错时打印行号与内容预览，按提示定位并修复/重传数据即可。

**Q: 无法从外网访问 TensorBoard 端口？**
A: 除服务器本机防火墙（如 `ufw`）外，云厂商或机房还可能有机柜/安全组策略。更稳妥做法是用 **SSH 本地端口转发**：`ssh -L 6006:127.0.0.1:6006 -p <端口> <用户>@<主机>`，浏览器访问本机 `http://127.0.0.1:6006`。

---

### 原理速览（帮你理解“为什么要这么做”）

- **Git LFS 指针文件**：仓库里大文件（数据集/模型等）用 LFS 管理，普通 `git clone` 只会拿到“指针文件”，必须 `git lfs pull` 才会拉到真实内容
- **pip 下载超时**：国内网络直连国外 PyPI 容易超时；升级 pip + 设置国内镜像，可以减少下载被重定向到国外源
- **模型软链接对齐**：脚本写死读取 `models/Qwen2.5-Coder-7B-Instruct`；把模型放在数据盘时，用软链接既省系统盘又不改代码
- **tmux 保活**：训练/评测是长任务，tmux 会话不受 SSH 断开影响

### 注意事项

1. **操作系统**：必须在 Linux 上运行（vLLM 不支持 Windows / macOS），推荐使用 Ubuntu 22.04 / 20.04
2. **模型路径硬编码**：`models/Qwen2.5-Coder-7B-Instruct` 在多个脚本中硬编码，请确保模型在此路径下（或通过软链接指向大盘上的实际目录）
3. **`latest_lora_adapter.txt`**：训练完成后自动写入适配器路径，LoRA 评测脚本读取此文件
4. **数据验证耗时**：`convet_KodCode_to_jsonl.py` 会 exec 每条代码做正确性验证，10,000 条可能需要较长时间
5. **磁盘空间**：模型约 14GB + 训练输出约 1–2GB，建议预留 ≥100GB 可用空间用于数据与 checkpoint
6. **计费与资源**：若使用按量计费 GPU，任务结束后及时关机或释放实例；重要结果请先 `scp` 或推送到仓库/对象存储备份
7. **网络访问**：SSH 端口可能非 22（如 `23822`）；对外暴露 TensorBoard 优先使用 SSH 隧道，见上文常见问题
