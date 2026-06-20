# Linux for AI

> 大多数 AI 都跑在 Linux 上。你需要懂到足够不会卡住。

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~30 minutes

## 学习目标

- 从命令行导航 Linux 文件系统，并完成基本文件操作
- 用 `chmod` 和 `chown` 管理文件权限，解决 “Permission denied” 错误
- 用 `apt` 安装系统 packages，并为 AI work 设置一台新的 GPU box
- 识别从 macOS 切到 Linux 时，远程机器开发中最容易踩的差异

## 问题

你在 macOS 或 Windows 上开发。但只要你 SSH 进一台 cloud GPU box、租一个 Lambda instance，或启动一台 EC2 machine，你就会落到 Ubuntu 里。Terminal 是你唯一的 interface。没有 Finder，没有 Explorer，没有 GUI。如果你不能从命令行导航文件系统、安装 packages、管理 processes，你就会一边为闲置 GPU 小时付费，一边搜索“how to unzip a file in Linux”。

这是一份生存指南。它只覆盖你在远程 Linux 机器上做 AI work 所需的内容。不多不少。

## File System Layout

Linux 把所有东西组织在单一 root `/` 下。没有 `C:\`，也没有 `/Volumes`。你真正会碰到的 directories：

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]
```

你的 home directory 是 `~` 或 `/home/your-username`。几乎所有事情都发生在这里。

## Essential Commands

下面这 15 个 commands 覆盖了你在远程 GPU box 上 95% 的操作。

### Moving Around

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### Files and Directories

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf` 是永久删除。没有 undo。按回车前一定再检查一次路径。

### Reading Files

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### Searching

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## Permissions

Linux 中每个文件都有 owner 和 permission bits。当 scripts 不能执行，或你无法写入某个 directory 时，你就会遇到它。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

常见修复：

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

当某个东西提示 “Permission denied”，几乎总是权限问题。`chmod +x` 或 `sudo` 能修好大多数情况。

## Package Management (apt)

Ubuntu 使用 `apt`。这是你安装 system-level software 的方式。

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

在新的 GPU box 上常安装的 packages：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## Users and sudo

你通常会以普通用户登录。有些操作需要 root（admin）权限。

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

在 cloud GPU instances 上，你通常是唯一用户，而且已经有 sudo access。不要把所有东西都用 root 跑。只有需要时才用 sudo。

## Processes and systemd

当 training 卡住，或你需要查看正在运行的东西：

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd 管理 services（background daemons）。如果你运行 inference servers，就会用到它：

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## Disk Space

GPU boxes 的 disk space 常常有限。Models 和 datasets 会很快把它填满。

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常见节省空间方式：

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## Networking

你会从命令行下载 models、传输 files、访问 APIs。

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

任何大文件都优先用 `rsync`，而不是 `scp`。它只传输变化的 bytes，并且能处理连接中断。

## tmux: Keep Sessions Alive

当你 SSH 到远程机器时，合上笔记本会杀掉 training run。tmux 可以防止这一点。

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

长时间 training jobs 永远放在 tmux 里跑。永远。

## WSL2 for Windows Users

如果你在 Windows 上，WSL2 可以给你一个真实 Linux environment，不需要 dual-boot。

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2 运行真实 Linux kernel。本课所有内容都能在里面使用。从 WSL 内部看，你的 Windows files 位于 `/mnt/c/Users/YourName/`。

GPU passthrough 在 Windows 侧安装 NVIDIA drivers 后可用。安装 Windows NVIDIA driver（不是 Linux 版），CUDA 就能在 WSL2 中使用。

## Gotchas: macOS to Linux

如果你从 macOS 过来，下面这些会绊倒你：

| macOS | Linux | Notes |
|-------|-------|-------|
| `brew install` | `sudo apt install` | Package names 有时不同。`brew install htop` 和 `sudo apt install htop` 作用一样，但 `brew install readline` 和 `sudo apt install libreadline-dev` 不一样。 |
| `open file.txt` | `xdg-open file.txt` | 但远程 box 上通常没有 GUI。用 `cat` 或 `less`。 |
| `pbcopy` / `pbpaste` | Not available | SSH 里没有 pipe to/from clipboard。 |
| `~/.zshrc` | `~/.bashrc` | macOS 默认 zsh。多数 Linux servers 用 bash。 |
| `/opt/homebrew/` | `/usr/bin/`, `/usr/local/bin/` | Binaries 位于不同位置。 |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS sed 在 `-i` 后需要 empty string。Linux 不需要。 |
| Case-insensitive filesystem | Case-sensitive filesystem | 在 Linux 上，`Model.py` 和 `model.py` 是两个不同文件。 |
| Line endings `\n` | Line endings `\n` | 相同。但 Windows 使用 `\r\n`，会破坏 bash scripts。用 `dos2unix` 修复。 |

## Quick Reference Card

```text
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

## Exercises

1. SSH 到任意 Linux machine（或打开 WSL2），并导航到你的 home directory。创建一个 project folder，在里面用 `touch` 创建三个空文件，然后用 `ls -la` 列出它们。
2. 用 apt 安装 `htop`，运行它，并识别哪个 process 使用了最多 memory。
3. 启动一个 tmux session，在里面运行 `sleep 300`，detach，列出 sessions，再 reattach。
4. 用 `df -h` 检查可用 disk space，然后用 `du -sh ~/.cache/*` 找出 cache 里占空间的内容。
5. 用 `scp` 从本地机器传一个文件到远程机器，然后用 `rsync` 做同样的传输，并比较体验。
