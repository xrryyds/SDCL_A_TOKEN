#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import logging
from huggingface_hub import snapshot_download

# =====================================================
# Logger
# # =====================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )/
# logger = logging.getLogger(__name__)

# MY_TOKEN = ""

# current_file_path = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(current_file_path))
# model_dir = os.path.join(project_root, "CELPO", "model", "OREAL")
# logger.info("downloading...")
# snapshot_download(
#     repo_id="internlm/OREAL-32B",
#     local_dir= os.path.join(model_dir, "OREAL-32B"),
#     token=MY_TOKEN,
#     # local_dir_use_symlinks=False
# )


import os
from huggingface_hub import snapshot_download

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))

# # 原 DS 下载, 已下完, 注释掉
# save_dir = os.path.join(
#     project_root, "SDCL_A_TOKEN", "model", "DS", "DeepSeek-R1-Distill-Qwen-7B"
# )
#
# print("...")
# snapshot_download(
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
#     cache_dir=None,
#     local_dir=save_dir,
#     revision="master",
# )
# print("")

# 2026-06-02 新增: Qwen3-8B (默认 thinking on, 跟 R1-Distill 同性质方便对比首 token 分布)
# 走 hf-mirror (顶部已设 HF_ENDPOINT), 避开 modelscope 代理问题
save_dir = os.path.join(
    project_root, "SDCL_A_TOKEN", "model", "Qwen", "Qwen3-8B"
)

print("...")
snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir=save_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("")


