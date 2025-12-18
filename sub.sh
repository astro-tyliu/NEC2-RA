#!/bin/bash
#SBATCH --job-name=pynec_sim        # 作业名称
#SBATCH --output=logs/%x_%j.out     # 标准输出文件
#SBATCH --error=logs/%x_%j.err      # 错误输出文件
#SBATCH --time=30:00:00             # 最大运行时间 30 小时
#SBATCH --cpus-per-task=1           # 使用 CPU 数量
#SBATCH --mem=8G                    # 内存（可根据需要调整）
#SBATCH --exclude=compute-103

# 1. 加载必要模块（如果有）
# module load python/3.10

# 2. 激活虚拟环境
source ~/venv/other/bin/activate

# 3. 创建日志目录（可选）
mkdir -p logs

# 4. 运行脚本
python /users/liutianyang/projects/other/nec2-ra/Tianyang.py
