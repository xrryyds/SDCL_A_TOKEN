# a_token_sdcl 运行命令参考

> 每段命令都是独立可复制的,改 §0 顶部变量即可适配不同实验。
> **当前默认全链路 max_token=4096**(数据池 / fill / 训练 answer / 推理生成 / vLLM max_model_len 全部统一)。
> 如需回退 2048,所有 `4096` 默认值都可命令行覆盖,见各 §。

---

## 完整执行顺序(从头到尾)

```
①  Stage 1+2  fill + merge          (1 条命令,单进程多卡 vLLM)
       ↓ 产出 a_token_train_data.json
②  Stage 3    DDP 训练               (1 条命令,torchrun-style 3 进程 DDP)
       ↓ 产出 LoRA checkpoint
③  Stage 4a   baseline 评测           (只需跑一次,一劳永逸)
       ↓ 产出 baseline 准确率
④  Stage 4b   LoRA 评测              (每次训完都跑,与 baseline 对比)
       ↓ 产出 mistake 救回率 / corr 保留率 / 全量准确率
```

每个阶段对应下面的章节,**第一次跑就按 §0 → §1 → §2 → §3.1 → §3.2 顺序**:

| 步骤            | 章节 | 命令片段                                                                    | 大概耗时 |
| --------------- | ---- | --------------------------------------------------------------------------- | -------- |
| 配置环境变量    | §0   | `export MODEL_PATH=... export CUDA_VISIBLE_DEVICES=...`                     | < 1 分钟 |
| ① fill + merge  | §1   | `python main.py pipeline --skip-train --fill_epoch $FILL_EPOCH`             | ~30 分钟 |
| ② DDP 训练      | §2   | `python scripts/train/run_a_token_sdcl_train.py ...`                        | ~40 分钟 |
| ③ baseline 评测 | §3.1 | `python main.py eval --model_path $MODEL_PATH`                              | ~10 分钟 |
| ④ LoRA 评测     | §3.2 | `python main.py eval --model_path $MODEL_PATH --adapter_path $ADAPTER_PATH` | ~10 分钟 |

> 想一条命令串完所有步骤,直接看 §4。

> 后续重跑实验时:
>
> - 若只换训练超参 → 跳过 §1,从 §2 开始
> - 若已有 fill_correct.json 想只重 merge → 用 `--skip-fill --skip-train`
> - baseline 跑过一次后就不用再跑 §3.1

---

## 0. 公共变量(每个实验改这里)

```bash
# 工程根目录
export PROJECT_ROOT=/workspace/SDCL_A_TOKEN
cd $PROJECT_ROOT

# 基座模型(教师 + 学生 base 都用这个)
export MODEL_PATH=/workspace/SDCL_A_TOKEN/model/DS/DeepSeek-R1-Distill-Qwen-7B

# 本次跑的 GPU(逗号分隔;若 dev3 被另一个实验占,这里就保持 0,1,2)
export CUDA_VISIBLE_DEVICES=0,1,2

# 训练输出目录(打时间戳,避免覆盖前一次实验)
export TRAIN_OUTPUT_DIR=output/a_token_sdcl_ddp_$(date +%Y%m%d_%H%M%S)

# 评测时用到的 LoRA checkpoint(训完再填,见 §3.2)
# 例:  export ADAPTER_PATH=$TRAIN_OUTPUT_DIR/checkpoint-1530
export ADAPTER_PATH=

# 训练数据(默认即可,自动产出)
export TRAIN_DATA_PATH=datasets/exam/a_token_train_data.json

# fill 阶段最大轮次(每轮在未救回题上换 seed 重 roll;连续 2 轮零新增提前停)
export FILL_EPOCH=10

# 错题池(默认走 4096 版本,跟代码默认一致)
# corr_answer_4096.json 已是默认,不需要传
export MISTAKE_PATH=datasets/exam/mistake_collection_book_4096.json
```

> 把这段 export 一次性粘进 shell 之后,后面所有命令都能直接复制运行。

---

## 1. Stage 1+2:fill + merge(产出 a_token_train_data.json)

```bash
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main.py pipeline \
    --mistake_path $MISTAKE_PATH \
    --skip-train \
    --fill_epoch $FILL_EPOCH
```

- 产物:
  - `datasets/exam/fill_correct.json`(填首 token 后能救回的题,**多轮累积并集**)
  - `datasets/exam/a_token_train_data.json`(corr_answer_4096 + fill_correct 合并)
  - `datasets/exam/_fill_rounds_tmp/`(每轮临时 mistake 子集 + fill 输出,事后可删)
  - `output/a_token_sdcl_<ts>/pipeline_dataflow.log` (数据流日志,含每轮 gain / pending)
  - `output/a_token_sdcl_<ts>/pipeline_samples.log` (样例日志)

> 长度参数(均为 4096 默认):
>
> - `--fill_max_gen_token 4096` fill 阶段每条续写最大 token 数
> - `--fill_prompt_len 1024` fill 阶段 prompt 上限
> - vLLM 实际 max_model_len = 1024 + 4096 = 5120
> - fill OOM 时:把 `--fill_prompt_len` 降到 768/512(4096 续写不需要太长 prompt)

> `--fill_epoch` 行为:
>
> - 第 1 轮跑完整 mistake 集;第 r 轮(r≥2)只跑前 r-1 轮"未救回"的题
> - 每轮 seed = `base_seed + (round-1)`,保证 roll 不同
> - 连续 2 轮新增 = 0 自动提前停;全部救回也提前停
> - `--fill_epoch 1` 等价于旧的"一次性"行为
>
> 已经有 `a_token_train_data.json` 时这步可以跳过,直接进 Stage 3。
> 已经有 `fill_correct.json` 想只重跑 merge:`python main.py pipeline --skip-fill --skip-train`

---

## 2. Stage 3:DDP 训练(榨满多卡)

```bash
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python scripts/train/run_a_token_sdcl_train.py \
    --model_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --output_dir $TRAIN_OUTPUT_DIR \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --ce_weight 1.0 \
    --no-gradient_checkpointing
```

要点:

- launcher 自动按 `CUDA_VISIBLE_DEVICES` 决定 `world_size`,3 卡 → 3 进程 DDP。
- **训练序列上限**:`max_prompt_length(1024) + max_answer_length(4096) = 5120` token,比 2048 时代多 67% 显存。
- **4096 时代如遇 OOM**:先去掉 `--no-gradient_checkpointing`(显存掉到 ~60GB);再不够把 `--batch_size` 降到 2;最后再考虑 `--max_answer_length 3072`。
- 端口冲突时加 `--master_port 29501`。
- **绝对不要**用 `python main.py` 跑训练,那是单进程入口,不会启动 DDP。

启动 30 秒后看 `$TRAIN_OUTPUT_DIR/train.log`,必须出现 `DDP 模式：world_size=3` 才算正确。

训练后的 checkpoint 在 `$TRAIN_OUTPUT_DIR/checkpoint-<step>/`。

---

## 3. Stage 4:评测(三集合一次出)

> 评测复用 `take_exam` 多卡 vLLM,**单次加载 + 单次推理**,持续吃满 3 卡。

### 3.1 baseline(初始模型,不带 LoRA;只需跑一次)

```bash
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main.py eval \
    --model_path $MODEL_PATH
```

输出示例:

```
📊 EVAL SUMMARY  (BASELINE)
  mistake : 0/3560 = 0.00%
  corr    : 3936/3936 = 100.00%
  all     : 3936/7496 = 52.51%
```

> baseline 数字以后所有实验都拿来对比,不变就只跑一次。

### 3.2 LoRA(训练后的 adapter)

跑前先填好 `ADAPTER_PATH`:

```bash
export ADAPTER_PATH=$TRAIN_OUTPUT_DIR/checkpoint-1530   # 改成实际目录
```

```bash
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main.py eval \
    --model_path $MODEL_PATH \
    --adapter_path $ADAPTER_PATH
```

输出示例:

```
📊 EVAL SUMMARY  (LoRA)
  mistake : 1850/3560 = 51.97%   ← 纠错率(越高越好)
  corr    : 3700/3936 = 94.00%   ← 保留率(应接近 baseline,掉太多说明蒸馏过头)
  all     : 5550/7496 = 74.04%   ← 综合
```

落盘目录:`output/eval_lora_<ts>/`

- `summary.json`
- `items_mistake.jsonl` / `items_corr.jsonl` / `items_all.jsonl`(题级 is_correct)

---

## 4. 一键全跑(从头到尾)

### 4.1 推荐:`pipeline_and_train`(数据 + 训练串成一条命令,异常/完成自动挂卡)

> 关键:不论中途 OOM、训练崩、还是顺利跑完,**main.py 的 finally 块都会进 `use_worker()` 挂卡保活**,GPU 不会被释放给其他人,人工接管后 Ctrl+C 退出即可。

```bash
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main.py pipeline_and_train \
    --mistake_path $MISTAKE_PATH \
    --fill_epoch $FILL_EPOCH \
    --model_path $MODEL_PATH \
    --train_output_dir $TRAIN_OUTPUT_DIR \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --ce_weight 1.0
```

补充透传给训练脚本的额外参数(如 `--no_ema`、`--max_answer_length 3072`):

```bash
    --train_extra_args "--no_ema --max_answer_length 3072"
```

OOM 时把上面去掉的 `--no-gradient_checkpointing`(默认开)用 `--gradient_checkpointing` 显式打开。

跑完后再单独跑评测(§3.2)即可。

### 4.2 分步跑(老方式,数据 / 训练分开,失败时不会自动挂卡)

```bash
# Stage 1+2
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py pipeline \
    --mistake_path $MISTAKE_PATH --skip-train --fill_epoch $FILL_EPOCH

# Stage 3
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python scripts/train/run_a_token_sdcl_train.py \
    --model_path $MODEL_PATH \
    --data_path $TRAIN_DATA_PATH \
    --output_dir $TRAIN_OUTPUT_DIR \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 --ce_weight 1.0 --no-gradient_checkpointing

# Stage 4 baseline(已跑过可跳过)
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py eval --model_path $MODEL_PATH

# Stage 4 LoRA(取最后一个 checkpoint)
LATEST_CKPT=$(ls -d $TRAIN_OUTPUT_DIR/checkpoint-* | sort -V | tail -1)
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main.py eval \
    --model_path $MODEL_PATH \
    --adapter_path $LATEST_CKPT
```

---

## 5. 常见排查

| 现象                             | 原因 / 处理                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| nvidia-smi 显示只有 dev0 高 util | 跑成了单进程旧版,没启动 DDP。检查日志开头有没有 `DDP 模式` 字样;清 `__pycache__` 重启     |
| OOM                              | 序列长 5120 比 2048 时代多 67% 显存。开 `gradient_checkpointing`,或 `--batch_size` 降到 2,或 `--max_answer_length 3072` |
| 评测准确率与训练 loss 对不上     | 评测脚本判分用的是 `extract_boxed_content + normalize_answer`,确认模型生成里有 `\boxed{}` |
| 端口被占                         | 训练命令加 `--master_port 29501`                                                          |
| 只想跑 baseline / 只想跑 LoRA    | Stage 4 两条命令是独立的,跑哪条就只看哪条                                                 |

---

## 6. 文件 / 目录索引

| 路径                                          | 含义                               |
| --------------------------------------------- | ---------------------------------- |
| `main.py`                                     | pipeline + eval 入口(子命令分发)   |
| `scripts/train/run_a_token_sdcl_train.py`     | DDP launcher(走 mp.spawn)          |
| `scripts/train/a_token_sdcl_train.py`         | DDP 训练主体                       |
| `scripts/inference/take_exam.py`              | vLLM 多卡推理(评测复用)            |
| `datasets/exam/mistake_DS_MATH.json`          | 评测:模型原本错的题(eval 默认)     |
| `datasets/exam/mistake_collection_book_4096.json` | 训练 fill 输入(4096 时代错题池) |
| `datasets/exam/corr_answer_4096.json`         | 训练 corr 输入 / 评测对照集(默认)  |
| `datasets/exam/a_token_train_data.json`       | 训练数据(corr_4096 + fill_correct 合并) |
| `output/a_token_sdcl_ddp_<ts>/`               | 训练产物(checkpoint + train.log)   |
| `output/eval_<lora\|baseline>_<ts>/`          | 评测产物(summary + items)          |
