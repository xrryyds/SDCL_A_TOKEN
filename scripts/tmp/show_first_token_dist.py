"""打印 eval_v3 输出目录下的首 token 分布 top20。

用法:
  python scripts/tmp/show_first_token_dist.py <eval_dir>
  python scripts/tmp/show_first_token_dist.py output/eval_v3_20260607_131236

可选: 通过 --tags 指定具体哪些 (默认: 全部 .json)
  python scripts/tmp/show_first_token_dist.py output/eval_v3_20260607_131236 \\
      --tags mistake_base_greedy mistake_lora_greedy
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=str, help="eval_v3 输出目录")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="指定 tag 列表 (e.g., mistake_lora_greedy); 默认全部")
    ap.add_argument("--top_n", type=int, default=20)
    args = ap.parse_args()

    if args.tags:
        files = []
        for tag in args.tags:
            f = os.path.join(args.eval_dir, f"first_token_dist_{tag}.json")
            if os.path.exists(f):
                files.append((tag, f))
            else:
                print(f"⚠ 文件不存在: {f}")
    else:
        files = []
        for f in sorted(glob.glob(os.path.join(args.eval_dir, "first_token_dist_*.json"))):
            tag = os.path.basename(f).replace("first_token_dist_", "").replace(".json", "")
            files.append((tag, f))

    if not files:
        print("没找到任何首 token 分布文件")
        return

    for tag, f in files:
        d = json.load(open(f))
        print(f"\n=== {tag}  (n_samples={d['n_samples']}) ===")
        print(f"  {'token_id':>8}  {'token_text':<25}  {'count':>8}  {'pct':>7}")
        for r in d["top20"][:args.top_n]:
            ttxt = r["token_text"][:25].replace("\n", "\\n")
            print(f"  {r['token_id']:>8}  {ttxt!r:<25}  {r['count']:>8}  {r['pct']:>6.2f}%")


if __name__ == "__main__":
    main()
