import json
from pathlib import Path

p = Path("demos/finance_demo/results/hard_eval")


def load(n):
    return json.loads((p / n).read_text(encoding="utf-8"))


def row(d, cfg):
    f = d["full_set_metrics"][cfg]
    r = d["recovery_on_dense_misses"][cfg]
    return (
        round(f["recall@10"], 3),
        round(f["MRR"], 3),
        r["recovered@10"],
        r.get("recovered@1"),
        round(f.get("recall@1", 0), 3),
    )


pre, now = load("multichannel_recovery.pre_bugfix.json"), load("multichannel_recovery.json")
print("MULTICHANNEL (pre -> now)")
print(f"{'config':22s} {'R@10':>12s} {'MRR':>12s} {'rec@10':>10s} {'rec@1':>10s} note")
for cfg in pre["full_set_metrics"]:
    a, b = row(pre, cfg), row(now, cfg)
    note = "SAME" if a == b else "DELTA"
    print(
        f"{cfg:22s} {a[0]}->{b[0]:<6} {a[1]}->{b[1]:<6} {a[2]}->{b[2]:<4} {a[3]}->{b[3]:<4} {note}"
    )

pp, pn = load("post_validation.pre_bugfix.json"), load("post_validation.json")
print("\nPOST-VALIDATION density")
for x, y in zip(pp["density"], pn["density"]):
    print(
        f"  noise={x['noise_edges']}: recovered@10 {x['recovered@10']}->{y['recovered@10']}  "
        f"P@10 {round(x['P@10'],3)}->{round(y['P@10'],3)}"
    )
print("POST OOD multi R@10")
for name in pp["ood"]:
    a, b = pp["ood"][name]["multi"]["recall@10"], pn["ood"][name]["multi"]["recall@10"]
    print(f"  {name}: {round(a,3)}->{round(b,3)}")

rp, rn = load("robustness_validation.pre_bugfix.json"), load("robustness_validation.json")
print("\nROBUSTNESS FP recovered@10 / MRR")
for cfg in rp["precision"]:
    a, b = rp["precision"][cfg], rn["precision"][cfg]
    print(
        f"  {cfg}: rec {a['recovered@10']}->{b['recovered@10']}  "
        f"MRR {round(a['MRR'],3)}->{round(b['MRR'],3)}  "
        f"xfp {round(a['cross_cluster_fp@10'],3)}->{round(b['cross_cluster_fp@10'],3)}"
    )
print("ROBUST sparsity recovered@10")
for a, b in zip(rp["sparsity"], rn["sparsity"]):
    print(f"  drop={a['drop_pct']}%: {a['recovered@10']}->{b['recovered@10']}")
print("ROBUST scale")
for cfg in rp["scale"]:
    a, b = rp["scale"][cfg], rn["scale"][cfg]
    print(
        f"  {cfg}: rec {a['recovered@10']}->{b['recovered@10']}  "
        f"R@10 {round(a['recall@10'],3)}->{round(b['recall@10'],3)}  "
        f"ms {round(a['mean_total_ms'],2)}->{round(b['mean_total_ms'],2)}"
    )
