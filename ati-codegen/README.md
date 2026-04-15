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

本项目需要 **Linux + NVIDIA GPU** 环境。使用 **阿里云 ECS GPU 实例** 作为远程训练/评测服务器，使用 **本地 Windows 电脑上的 WSL 或虚拟机** 作为 SSH 客户端连接和操作云服务器。


| 项目      | 要求                                              |
| ------- | ----------------------------------------------- |
| 云服务器 OS | Linux（阿里云 ECS 选择 Ubuntu 22.04 / 20.04）          |
| GPU     | NVIDIA，≥16GB 显存（推荐 24GB+）                       |
| CUDA    | 11.8 或 12.x                                     |
| Python  | 3.10+                                           |
| 磁盘      | 系统盘 ≥40GB + 数据盘 ≥100GB（模型 14GB + 数据 + 训练输出）     |
| 本地电脑    | Windows 10/11 + WSL2 或 Linux 虚拟机（用于 SSH 连接云服务器） |


> **阿里云 GPU 实例规格选择**：
>
> - **预算充足**：ecs.gn7e-c16g1.4xlarge（A100-80G，训练 + 评测全程无压力）
> - **性价比首选**：ecs.gn7i-c16g1.4xlarge（A10-24G，完全够用）
> - **最低门槛**：ecs.gn6v-c8g1.2xlarge（V100-16G，需调低 batch size，详见「显存调优」章节）
>
> 也可使用**抢占式实例**（竞价实例），价格约为按量付费的 10%-20%，适合可中断的训练任务。

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
2. **通过 SSH 连接到阿里云 GPU 服务器**（在本地 WSL 或虚拟机的 Linux 终端中操作）
3. **将代码上传到云服务器**（通过 git clone、scp 等方式）
4. **远程执行训练和评测**（所有计算在云服务器的 GPU 上完成）
5. **下载结果到本地**（评测结果、训练日志等）

```text
┌──────────────────────┐         SSH          ┌────────────────────────────┐
│  Windows 本地电脑     │ ──────────────────→  │  阿里云 ECS GPU 服务器      │
│                      │                      │                            │
│  ┌────────────────┐  │   上传代码 (scp/git)  │  Ubuntu 22.04              │
│  │ WSL2 / 虚拟机   │  │ ──────────────────→  │  NVIDIA GPU (A10/V100/A100)│
│  │ (SSH 客户端)    │  │                      │  PyTorch + vLLM + PEFT     │
│  └────────────────┘  │   下载结果 (scp)      │                            │
│                      │ ←──────────────────   │  训练 → 评测 → 结果输出     │
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

### 阿里云 GPU 环境配置（完整教程）

以下教程从 **本地 Windows 电脑（通过 WSL / 虚拟机的 Linux 终端）** 出发，完成阿里云 GPU 实例的创建、连接、环境搭建和代码运行全流程。

#### 第一步：创建阿里云 GPU 实例

1. 登录 [阿里云 ECS 控制台](https://ecs.console.aliyun.com/)
2. 点击「创建实例」，选择地域（推荐华东/华北/华南，GPU 库存较充足）
3. **实例规格**：选择 GPU 计算型（参考上方实例规格选择），搜索 `gn7i`、`gn7e` 或 `gn6v` 系列
4. **镜像选择**（关键）：
  - 进入「镜像市场」，搜索 **NVIDIA GPU Cloud VM Image** 或 **深度学习镜像**
  - 推荐选择阿里云官方提供的 **深度学习镜像（Ubuntu 22.04）**，已预装 NVIDIA 驱动、CUDA、cuDNN、PyTorch 等
  - 也可选择 Ubuntu 22.04 公共镜像，后续手动安装
5. **存储**：系统盘 40GB SSD + 数据盘 100GB+ ESSD（用于存放模型和训练输出）
6. **网络**：分配公网 IP（按流量计费），或后续通过 EIP 绑定
7. **安全组**：放行 22 端口（SSH）、6006 端口（TensorBoard，可选）
8. 设置 root 密码或 SSH 密钥对，确认创建

#### 第二步：从本地 Linux 环境连接到阿里云服务器

打开本地 **WSL 终端**（或虚拟机终端），通过 SSH 连接云服务器：

```bash
# ===== 在本地 WSL / 虚拟机终端中执行 =====

# SSH 连接（替换 <公网IP> 为你的阿里云 ECS 公网 IP）
ssh root@<公网IP>
# 首次连接会提示确认指纹，输入 yes
# 然后输入创建实例时设置的 root 密码

# 连接成功后，你已经在阿里云服务器的终端中了
# 确认 GPU 可用
nvidia-smi
```

确认输出中有 GPU 信息（型号、显存、CUDA 版本）。如果使用阿里云深度学习镜像，驱动已预装，这一步应该直接通过。

> **SSH 免密登录（可选，推荐配置）**：
> 每次输入密码比较麻烦，可以配置 SSH 密钥免密登录：
>
> ```bash
> # ===== 在本地 WSL / 虚拟机终端中执行 =====
>
> # 生成 SSH 密钥（如果已有 ~/.ssh/id_rsa 则跳过）
> ssh-keygen -t rsa -b 4096
> # 一路回车使用默认值
>
> # 将公钥上传到云服务器
> ssh-copy-id root@<公网IP>
>
> # 之后连接就不用输密码了
> ssh root@<公网IP>
> ```

> **如果 `nvidia-smi` 报错**：说明选择的是公共镜像且未安装驱动，需手动安装：
>
> ```bash
> # 在云服务器上执行
> sudo apt update && sudo apt install -y nvidia-driver-535
> sudo reboot
> # 重启后重新 SSH 连接
> ```

#### 第三步：挂载数据盘 & 上传项目代码

以下操作需要在两个不同的终端环境中执行，请注意区分：

**3.1 在云服务器上挂载数据盘**（SSH 连上后执行）：

```bash
# ===== 在阿里云服务器终端中执行 =====

# 查看数据盘设备名（通常为 /dev/vdb）
lsblk

# 格式化（仅首次，如已格式化请跳过）
mkfs.ext4 /dev/vdb

# 挂载到 /mnt/data
mkdir -p /mnt/data
mount /dev/vdb /mnt/data

# 设置开机自动挂载
echo '/dev/vdb /mnt/data ext4 defaults 0 0' >> /etc/fstab
```

**3.2 将项目代码上传到云服务器**：

```bash
# ===== 方式一：在云服务器上 git clone（强烈推荐，最省事）=====
# 这是最推荐的方式：后续你在本地 push 更新后，只需在云服务器执行 git pull 即可同步，无需反复 scp 手动上传。
# 本项目仓库地址：
#   https://github.com/WilliamGuYZ/graduationDesign.git
#
# 注意：本仓库包含大文件数据（如 KodCode.jsonl），使用 Git LFS 管理。
# 如果不安装 git-lfs，云服务器上拿到的会是“指针文件”，第一行类似：
#   version https://git-lfs.github.com/spec/v1
# 这会导致 split_train_eval.py 读 JSONL 时报 JSONDecodeError。
#
# 在云服务器终端中执行：
cd /mnt/data
apt update && apt install -y git-lfs
git lfs install

git clone https://github.com/WilliamGuYZ/graduationDesign.git graduationDesign
cd graduationDesign/ati-codegen

# 拉取 LFS 大文件（关键一步）
git lfs pull

# ===== 方式二：从本地 WSL / 虚拟机通过 scp 上传 =====
# 新开一个本地 WSL 终端（不要在 SSH 会话中执行）：

# 如果项目在 Windows 桌面（WSL 中访问 Windows 路径需通过 /mnt/）
cd /mnt/d/My\ Documents/11191857/Desktop/gyz/graduationDesign
scp -r ati-codegen/ root@<公网IP>:/mnt/data/

# 上传完成后，在云服务器终端中：
cd /mnt/data/ati-codegen

# ===== 方式三：通过 OSS 中转（适合大文件）=====
# 先在本地打包上传到 OSS，再从 ECS 内网高速下载
# ossutil cp oss://<bucket>/ati-codegen.tar.gz /mnt/data/
```

> **提示**：强烈推荐方式一（git clone + git lfs pull）。方式二适合项目未推送到远程仓库，或仅临时传少量文件的情况。

#### 后续使用（已完成首次部署后）

当你在本地提交并推送了新代码后，在云服务器项目目录执行下面几条命令即可完成同步并重跑，无需再用 scp 手动上传：

```bash
cd /mnt/data/graduationDesign/ati-codegen
git pull
git lfs pull   # 只有仓库大文件更新时才需要
pip install -r requirements.txt   # 只有依赖变化时才需要

tmux new -s run
bash scripts/train_eval.sh
```

#### 第四步：升级 pip & 配置阿里云镜像

阿里云 ECS 公共镜像自带的 pip 版本较旧（22.x），对国内镜像的处理存在 bug，会导致下载仍跳转到国外源超时。**必须先升级 pip 并配置全局镜像**：

```bash
# 1. 升级 pip（旧版 pip 对镜像重定向有 bug，导致实际下载仍走国外源）
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --default-timeout=100

# 2. 设置全局 pip 配置，永久使用阿里云镜像（之后所有 pip 命令自动走阿里云，无需每次加 -i 参数）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

# 3. 验证配置生效
pip config list
# 应输出：
# global.index-url='https://mirrors.aliyun.com/pypi/simple/'
# global.trusted-host='mirrors.aliyun.com'
```

#### 第五步：安装依赖

```bash
# 如果使用了深度学习镜像自带的 conda 环境，先激活（可选）
# conda activate base

# 确认 Python 和 PyTorch 是否已预装
python3 --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

**如果上述命令输出正常**（PyTorch 已安装、CUDA 可用），直接安装其余依赖：

```bash
# 全局镜像已配置，直接 pip install 即可（自动走阿里云镜像）
pip install -r requirements.txt
```

**如果选择了公共镜像没有预装 PyTorch**，需要先手动安装（推荐按顺序执行，避免依赖下载走国外源导致超时）：

```bash
# 查看 CUDA 版本
nvidia-smi   # 右上角显示 CUDA Version

# 先安装 torch 的依赖（从阿里云镜像下载，很快）
pip install sympy networkx filelock

# 安装 PyTorch（torch 本体从 PyTorch 官方源下载，依赖已装好不会再下载）
# CUDA 12.x：
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8：
# pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

# 然后安装项目其余依赖
pip install -r requirements.txt
```

> **验证安装**：
>
> ```bash
> python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
> python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
> ```
>
> 两条命令都不报错即可。

#### 第六步：下载基座模型

将 Qwen2.5-Coder-7B-Instruct 模型下载到 `models/` 目录：

```bash
# 方式一：modelscope（阿里云内网下载极快，强烈推荐）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir models/Qwen2.5-Coder-7B-Instruct

# 方式二：huggingface-cli（阿里云 ECS 访问 huggingface 可能较慢，不推荐）
# pip install huggingface_hub
# huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir models/Qwen2.5-Coder-7B-Instruct
```

> 下载约 14GB。阿里云 ECS 通过 modelscope 下载通常几分钟即可完成。
> 下载完成后确认 `models/Qwen2.5-Coder-7B-Instruct/` 下有 `config.json`、`*.safetensors` 等文件。
>
> **磁盘空间提示**：建议将模型下载到数据盘，再做软链接到项目目录：
>
> ```bash
> # 模型下载到数据盘 /mnt/data/
> modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir /mnt/data/Qwen2.5-Coder-7B-Instruct
> ln -s /mnt/data/Qwen2.5-Coder-7B-Instruct models/Qwen2.5-Coder-7B-Instruct
> ```

> **路径对齐提示（非常重要）**：
> 训练/评测脚本默认使用项目内路径 `models/Qwen2.5-Coder-7B-Instruct`。如果你把模型下载在 `/root/models/...` 或 `/mnt/data/models/...`，必须通过软链接对齐到项目目录，否则 Transformers 会把这个路径当成 HuggingFace 仓库 id，报：
> `HFValidationError: Repo id must be in the form ...`
> ```bash
> cd /mnt/data/graduationDesign/ati-codegen
> mkdir -p models
> ln -s /root/models/Qwen2.5-Coder-7B-Instruct models/Qwen2.5-Coder-7B-Instruct
> # 或 ln -s /mnt/data/models/Qwen2.5-Coder-7B-Instruct models/Qwen2.5-Coder-7B-Instruct
> ```

#### 第七步：数据准备

> **如果仓库中已有 `data/raw/KodCode.jsonl` 和 `data/processed/KodCode_train.jsonl`、`KodCode_eval.jsonl`，可以跳过本步骤，直接进入第八步。**

```bash
# 0) 若你是用 git clone 拉的仓库：先确认 LFS 大文件已拉取（只需做一次）
# 如果输出第一行是 "version https://git-lfs.github.com/spec/v1"，说明还是指针文件，需要执行：
#   git lfs pull
head -n 1 data/raw/KodCode.jsonl

# 1. 如果只有 Parquet，先转换为 JSONL（会执行代码验证，耗时较长，约 30-60 分钟）
#    如果 data/raw/KodCode.jsonl 已存在则跳过此步
python3 scripts/convet_KodCode_to_jsonl.py

# 2. 划分训练/评测集（9:1）
#    如果 data/processed/ 下已有 KodCode_train.jsonl 和 KodCode_eval.jsonl 则跳过此步
python3 scripts/split_train_eval.py --ratio 0.9 --no-shuffle
```

#### 第八步：运行全流程

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

#### 第九步：查看结果

```bash
# 评测结果
cat evaluation/results/eval_base_passk.txt
cat evaluation/results/eval_lora_passk.txt

# TensorBoard 训练曲线（可选）
tensorboard --logdir train/outputs/ --bind_all
# 浏览器访问 http://<公网IP>:6006
# 注意：需在阿里云安全组中放行 6006 端口（入方向，TCP）
```

#### 第十步：下载结果到本地

训练和评测完成后，在本地 **WSL / 虚拟机终端**（不是 SSH 会话）中，将结果下载到本地电脑：

```bash
# ===== 在本地 WSL / 虚拟机终端中执行（不是 SSH 会话）=====
# 替换 <公网IP> 为你的实际公网 IP

# 先进入你想保存结果的目录（例如 Windows 桌面）
cd /mnt/d/My\ Documents/11191857/Desktop/gyz/graduationDesign/ati-codegen

# 下载评测结果
scp root@<公网IP>:/mnt/data/graduationDesign/ati-codegen/evaluation/results/*.txt ./evaluation/results/

# 下载训练好的 LoRA adapter（如需后续部署或展示）
scp -r root@<公网IP>:/mnt/data/graduationDesign/ati-codegen/train/outputs/ ./train/outputs/

# 下载 TensorBoard 日志（可选，用于本地查看训练曲线）
# scp -r root@<公网IP>:/mnt/data/graduationDesign/ati-codegen/train/outputs/*/runs/ ./tb_logs/
```

> **省钱提醒**：
>
> - 训练和评测完成后记得在 ECS 控制台**停止实例**！GPU 实例按时计费，停机后不再计算费用（仅收取少量云盘费用）
> - 如使用**抢占式实例**，注意保存中间结果（checkpoint），被回收后可快速恢复
> - 长期不用建议把结果下载到本地或上传到 OSS 后**释放实例**，避免持续产生云盘费用

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


> **各阿里云 GPU 实例参考配置**：
>
> - **gn7e（A100-80G）**：全部默认值即可
> - **gn7i（A10-24G）**：默认值即可，评测时 `GPU_MEMORY_UTILIZATION` 可降为 0.85
> - **gn6v（V100-16G）**：训练 `BATCH_SIZE=1, GRADIENT_ACCUMULATION=32, MAX_LENGTH=512`；评测 `GPU_MEMORY_UTILIZATION=0.7, VLLM_MAX_MODEL_LEN=1024`

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
A: 可以。Windows 10 1809+ 自带 OpenSSH 客户端，在 PowerShell 中直接执行 `ssh root@<公网IP>` 即可。但 scp 上传文件时路径处理不如 Linux 终端方便，且后续操作（如查看 man 手册、使用 rsync 等）在 Linux 终端体验更好，因此推荐 WSL。

#### 云服务器相关

**Q: `nvidia-smi` 找不到命令？**
A: NVIDIA 驱动未安装。检查是否选错了非 GPU 实例规格。如果选择的是公共镜像，需要手动安装驱动：`sudo apt install -y nvidia-driver-535 && sudo reboot`。使用阿里云深度学习镜像则不会出现此问题。

**Q: `torch.cuda.is_available()` 返回 `False`？**
A: PyTorch 的 CUDA 版本与系统不匹配。重新安装：`pip install torch --index-url https://download.pytorch.org/whl/cu121`。如果选了阿里云深度学习镜像，一般不会出现此问题。

**Q: `pip install torch` / `pip install sympy` 下载很慢或超时？**
A: 先按「第四步：升级 pip & 配置阿里云镜像」升级 pip 并设置全局镜像。旧版 pip（22.x）可能会把下载跳转到 `files.pythonhosted.org`（国外）导致超时。公共镜像安装 torch 建议按「第五步」先装 `sympy/networkx/filelock` 再装 torch。

**Q: `pip install vllm` 特别慢或失败？**
A: 使用阿里云 PyPI 镜像：`pip install vllm -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`。如果编译报错，确认 CUDA 版本 ≥ 11.8 且有 gcc（`sudo apt install -y build-essential`）。

**Q: `bash scripts/train_eval.sh` 报 `set: pipefail: invalid option name`？**
A: 脚本文件是 Windows 的 CRLF 换行导致的（实际变成 `pipefail\\r`）。在云服务器执行：
```bash
apt update && apt install -y dos2unix
dos2unix scripts/*.sh
```

**Q: SSH 断开后训练就停了？**
A: 长任务必须用 `tmux` 或 `nohup` 运行。参考第七步中的说明。即使关闭 WSL 终端或合上笔记本，云服务器上的 tmux 会话仍在运行。

**Q: 训练时 OOM？**
A: 降低 `train/train.py` 中的 `BATCH_SIZE`（如从 4 降为 2），同时增大 `GRADIENT_ACCUMULATION`（如从 8 增到 16）。参考「显存调优」章节中各显卡的参考配置。

**Q: 评测时 OOM？**
A: 降低评测脚本中的 `GPU_MEMORY_UTILIZATION`（如从 0.9 降为 0.7）和 `VLLM_MAX_MODEL_LEN`（如从 2048 降为 1024）。

**Q: `latest_lora_adapter.txt` 路径不对？**
A: 该文件在训练完成后自动生成，内容为 adapter 的绝对路径。如果把项目移到了新位置，需要手动修改文件内容指向正确路径，或重新训练。

**Q: 阿里云数据盘没有自动挂载？**
A: 阿里云 ECS 的数据盘需要手动挂载。参考第三步中的数据盘挂载操作，挂载到 `/mnt/data` 后即可使用。建议把模型和训练输出放在数据盘上。

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

**Q: 安全组如何配置？**
A: 在 ECS 控制台 → 实例 → 安全组 → 入方向规则中添加。必需放行 22/TCP（SSH），如需使用 TensorBoard 还需放行 6006/TCP。

---

### 原理速览（帮你理解“为什么要这么做”）

- **Git LFS 指针文件**：仓库里大文件（数据集/模型等）用 LFS 管理，普通 `git clone` 只会拿到“指针文件”，必须 `git lfs pull` 才会拉到真实内容
- **pip 下载超时**：国内云服务器直连国外 PyPI 容易超时；升级 pip + 设置国内镜像，可以避免下载跳转到国外源
- **模型软链接对齐**：脚本写死读取 `models/Qwen2.5-Coder-7B-Instruct`；把模型放在数据盘时，用软链接既省系统盘又不改代码
- **tmux 保活**：训练/评测是长任务，tmux 会话不受 SSH 断开影响

### 注意事项

1. **操作系统**：必须在 Linux 上运行（vLLM 不支持 Windows / macOS），阿里云 ECS 选择 Ubuntu 镜像即可
2. **模型路径硬编码**：`models/Qwen2.5-Coder-7B-Instruct` 在多个脚本中硬编码，请确保模型放在此路径下（或通过软链接指向数据盘上的实际位置）
3. `**latest_lora_adapter.txt`**：训练完成后自动写入适配器路径，LoRA 评测脚本读取此文件
4. **数据验证耗时**：`convet_KodCode_to_jsonl.py` 会 exec 每条代码做正确性验证，10,000 条可能需要较长时间
5. **磁盘空间**：模型约 14GB + 训练输出约 1-2GB，建议数据盘至少 100GB（系统盘 40GB + 数据盘 100GB）
6. **省钱**：阿里云 GPU 实例按量计费，完成后务必在 ECS 控制台停止实例；长时间不用建议把结果上传到 OSS 或下载到本地后释放实例
7. **安全组**：确保已在阿里云安全组放行 SSH（22 端口），TensorBoard 需额外放行 6006 端口

