# 使用 Miniconda 基础镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制环境配置文件
COPY environment.yml .

# 创建 conda 环境
RUN conda config --set remote_read_timeout_secs 60 && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda env create -f environment.yml


# 激活环境并设置 PATH
# 获取环境名称（environment.yml 第一行的名字）
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# 复制你的项目代码
COPY . .

# 设置默认命令（使用 conda run 来在指定环境中运行）
CMD ["conda", "run", "-n", "base", "python", "app.py"]