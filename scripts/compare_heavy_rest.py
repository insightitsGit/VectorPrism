import json
from pathlib import Path

p = Path("demos/finance_demo/results/hard_eval")


def load(n):
    return json.loads((p / n).read_text(encoding="utf-8"))


pp, pn = load("post_validation.pre_bugfix.json"), load("post_validation.json")
print("POST density levels (pre -> now)")
for a, b in zip(pp["density"]["levels"], pn["density"]["levels"]):
    noise = a.get("noise", a.get("noise_edges"))
    print(
        f"  noise={noise}: rec {a['recovered@10']}->{b['recovered@10']} "
        f"P@10 {round(a['P@10'], 3)}->{round(b['P@10'], 3)}"
    )

print("POST OOD multi recall@10")
for name, a in pp["ood"].items():
    b = pn["ood"][name]
    # tolerate key variants
    def multi_r10(x):
        if "multi_R@10" in x:
            return x["multi_R@10"]
        if "multi" in x and isinstance(x["multi"], dict):
            return x["multi"].get("recall@10", x["multi"].get("R@10"))
        return x.get("multi_recall@10")

    print(f"  {name}: {round(multi_r10(a), 3)}->{round(multi_r10(b), 3)}")
    if name == "easy_finance":
        print("   keys", list(a.keys()))

rp, rn = load("robustness_validation.pre_bugfix.json"), load("robustness_validation.json")
print("ROBUST precision_profile")
for k, a in rp["precision_profile"].items():
    b = rn["precision_profile"][k]
    print(
        f"  {k}: rec {a['recovered@10']}->{b['recovered@10']} "
        f"MRR {round(a['MRR'], 3)}->{round(b['MRR'], 3)}"
    )

print("ROBUST sparsity")
for a, b in zip(rp["sparsity"]["fractions"], rn["sparsity"]["fractions"]):
    print(f"  drop={a['drop_pct']}%: rec {a['recovered@10']}->{b['recovered@10']}")

print("ROBUST scale")
for k, a in rp["scale"]["configs"].items():
    b = rn["scale"]["configs"][k]
    print(
        f"  {k}: rec {a['recovered@10']}->{b['recovered@10']} "
        f"R@10 {round(a['recall@10'], 3)}->{round(b['recall@10'], 3)}"
    )
