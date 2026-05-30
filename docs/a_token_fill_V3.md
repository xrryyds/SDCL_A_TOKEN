# a_token_fill_V3 — 实验进度记录

## 1. Baseline 节点 (2026-05-30)

历史实验全部作废,从干净 baseline 重启。

模型:`DeepSeek-R1-Distill-Qwen-7B`
评测设置:MATH train 7496 题,greedy 单次,2048 prompt + 4096 gen

| 指标 | 数值 |
|---|---|
| acc | **72.92%** (5466/7496) |
| mistake 池 | 2030(实际盘上 `mistake_DS_MATH_pool.json` 为 **2040** 题)|
| corr 池 | 5466 |

数据文件:
- `datasets/exam/mistake_DS_MATH_pool.json` (2040 题)
- `datasets/exam/corr_DS_MATH_pool.json` (5466 题)
- `datasets/exam/mistake_collection_book_4096.json` (2030 题)
- `datasets/exam/corr_answer_4096.json` (5466 题)

---

## 2. 思路调整

原 `a_token_sdcl` 的 fill_correct 流程:**单 fill_token + 单 answer**(从 `first_tokens_test.json` 池随机抽 N 个首 token,强制塞 prompt + greedy 续写,挑首 token logprob 最大的 1 条做对答案保留)。

V3 调整为:**收集多个对答案的首 token 作为多候选**(同一题保留所有不同首 token 的对答案,做多 candidates 训练)。

---

## 3. fill_multi 数据采集

脚本:`scripts/build_fill_multi_pool.py`(DP 形态:每卡一个子进程 + `tensor_parallel_size=1` + `mp.spawn` + Queue)

### 算法

对 mistake 池每题,做最多 10 轮 rolling-K rollout:

- 每轮 K=16 条 rollout,T=0.6,top_p=0.95(**自由采样,无首 token 池注入**)
- 收集本轮中对的(boxed 字符串相等)
- 按 `first_token_id` 去重(每个 token id 保留首次出现的对答案)
- 该轮去重后所有 candidates 写入结果,题从 active 移除
- 否则保留 active,进下一轮

### 运行(4 卡 H800)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/build_fill_multi_pool.py --k 16 --max_rounds 10
```

### 结果

| 指标 | 数值 |
|---|---|
| 救回 | **1296 / 2040 = 63.53%** |
| unresolved | 744 / 2040 = 36.47% |
| 耗时 | 235.0 min(14100.2s) |
| candidates/题 | min=1, max=4, avg=1.65, median=2 |

**rounds_used 分布(救回轮次):**

| Round | 救回题数 |
|---|---|
| 1 | 1075 |
| 2 | 78 |
| 3 | 42 |
| 4 | 35 |
| 5 | 13 |
| 6 | 16 |
| 7 | 10 |
| 8 | 10 |
| 9 | 9 |
| 10 | 8 |

Round 1 救回 1075 / 1296 = 83%,后 9 轮边际递减明显。

### 产出文件

- `datasets/exam/fill_multi_pool.json` — 1296 题救回数据(每题 candidates 数组)
- `datasets/exam/fill_multi_unresolved.json` — 744 题未救回

每条 rescued entry 结构:
```json
{
  "question_idx": int,
  "question": str,
  "ref_answer": str,
  "candidates": [
    {"token_id": int, "token_text": str, "answer": str, "round": int},
    ...
  ],
  "n_rounds_used": int,
  "total_correct_rollouts_this_round": int,
  "total_rollouts": int,
  "source": "fill_multi"
}
```

---

## 4. 首 token 池整理

### 4.1 从 fill_multi 产出统计的首 token 池(新)

文件:`datasets/first_tokens_DS_roll.json`

来源:对 `fill_multi_pool.json` 里所有 candidates 按 `token_id` 去重 + count 累加。

| 字段 | 数值 |
|---|---|
| total_solutions | 2134(= 所有 candidates 总数) |
| unique_tokens | **4** |

| token_id | token_text | count |
|---|---|---|
| 32313 | `Okay` | 1135 |
| 71486 | `Alright` | 943 |
| 5338 | `First` | 39 |
| 1249 | `To` | 17 |

`Okay` + `Alright` 占 (1135+943)/2134 = **97.4%**。R1-Distill-Qwen-7B 在 T=0.6 自由采样下首 token 高度集中。

### 4.2 整理 `first_tokens_test.json`

- `datasets/first_tokens_test_origin.json` — 原版备份,**379 tokens**, total_solutions=12496
- `datasets/first_tokens_test.json` — 去掉上面 4 个 token(实际命中 3 个,`Alright` 不在原池),**376 tokens**, total_solutions=12496(未变)

被去掉的(原 test 池中的 count):
| token_id | token_text | original test count |
|---|---|---|
| 5338 | `First` | 451 |
| 1249 | `To` | 201 |
| 32313 | `Okay` | 1 |
| 71486 | `Alright` | (不在 test 池) |
