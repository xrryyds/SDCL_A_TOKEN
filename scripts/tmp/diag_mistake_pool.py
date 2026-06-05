"""诊断 mistake 池 Base 评测 27% 异常的根因。

mistake 池本应是 Base 判错的题, Base 重评该接近 0%, 但评出 27.20% (386/1419)。
本脚本不跑模型, 纯静态分析 mistake/corr/fill 池, 定位是"池脏了"还是"真翻盘"。

用法 (远程):
  cd /workspace/SDCL_A_TOKEN
  python scripts/tmp/diag_mistake_pool.py
"""
import json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.data_utils import extract_boxed_content, normalize_answer

EXAM = os.path.join(_ROOT, "datasets", "exam")
MIS = os.path.join(EXAM, "mistake_DS_MATH_pool.json")
CORR = os.path.join(EXAM, "corr_DS_MATH_pool.json")
FILL = os.path.join(EXAM, "fill_multi_pool.json")


def _load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_boxed(t):
    b = extract_boxed_content(t or "")
    return normalize_answer(b) if b is not None else None


print("=" * 70)
print("诊断 mistake 池 Base 27% 异常")
print("=" * 70)

mis = _load(MIS)
corr = _load(CORR)
print(f"\nmistake 池: {len(mis)} 题  | keys={list(mis[0].keys())}")
print(f"corr    池: {len(corr)} 题")

# ---- 1) mistake 池里存的 answer 的 boxed 是否真 ≠ ref ----
n_match = 0       # 存的 answer 居然判对 (池脏)
n_no_boxed = 0    # 存的 answer 根本没 boxed
n_real_wrong = 0  # 存的 answer 确实错 (正常错题)
for it in mis:
    b = _norm_boxed(it.get("answer", ""))
    ref = normalize_answer(str(it.get("ref_answer", "")))
    if b is None:
        n_no_boxed += 1
    elif b == ref:
        n_match += 1
    else:
        n_real_wrong += 1
N = len(mis)
print("\n--- 1) mistake 池里存的 answer (构造时 Base 输出) 判分复核 ---")
print(f"  真错 (boxed≠ref)        : {n_real_wrong}/{N} = {n_real_wrong/N*100:.1f}%  ← 应接近100%")
print(f"  没 boxed (截断/没答完)  : {n_no_boxed}/{N} = {n_no_boxed/N*100:.1f}%  ← 这些是'被截断判错'")
print(f"  居然判对 (boxed==ref)   : {n_match}/{N} = {n_match/N*100:.1f}%  ← 应=0%, 若高则池脏")

# ---- 2) mistake 与 corr 的 question_idx 是否重叠 (池构造 bug) ----
mis_idx = set(it.get("question_idx") for it in mis)
corr_idx = set(it.get("question_idx") for it in corr)
overlap = mis_idx & corr_idx
print("\n--- 2) mistake / corr question_idx 重叠检查 ---")
print(f"  mistake unique idx: {len(mis_idx)} | corr unique idx: {len(corr_idx)}")
print(f"  重叠 idx 数: {len(overlap)}  ← 应=0, 若>0 则同题既在corr又在mistake (池构造bug)")

# ---- 3) "没 boxed" 的占比是关键: 这些题是因截断判错, 评测窗口够大就可能做对 ----
print("\n--- 3) 解读 ---")
no_boxed_pct = n_no_boxed / N * 100
if n_match > N * 0.05:
    print(f"  ⚠ 池脏: {n_match} 题存的answer其实判对了, 不该在mistake池")
elif no_boxed_pct > 20:
    print(f"  ⚠ {no_boxed_pct:.0f}% 题是'没boxed'(构造时8192内没写完答案 → 判错)")
    print(f"    这些题评测时若同样没写完仍是错, 但 vLLM 重新生成时长链可能这次写完了 → Base翻盘做对")
    print(f"    这能解释 Base 在 mistake 上有正确率 (27% ≈ 这批边界题的翻盘率)")
else:
    print(f"  池基本干净 (真错{n_real_wrong/N*100:.0f}%), 27% 主要来自 vLLM greedy 重生成的非确定性")

# ---- 4) 抽 3 题看实例 ----
print("\n--- 4) 抽样 3 题 (mistake 池存的 answer 尾部 + ref) ---")
for it in mis[:3]:
    ans = it.get("answer", "")
    b = _norm_boxed(ans)
    print(f"\n  q_idx={it.get('question_idx')} ref={it.get('ref_answer')!r}")
    print(f"    存的answer boxed={b!r}  answer长度={len(ans)}字符")
    print(f"    answer尾100: {ans[-100:]!r}")

print("\n" + "=" * 70)
