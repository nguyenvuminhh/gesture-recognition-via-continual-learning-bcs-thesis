import pandas as pd
from matplotlib import pyplot as plt

names_by_method = {m: method_names[methods.index(m)] for m in methods}

acc_matrix = {c: {} for c in context_range}
for m, r in results.items():
    for c, acc in enumerate(r.context_accuracies, start=1):
        acc_matrix[c][m] = acc * 100.0

rank_rows = list(range(1, len(methods) + 1))
cols = [f"C{c}" for c in context_range]
cell_text = []

for rank in rank_rows:
    row = []
    for c in context_range:
        ranked = sorted(acc_matrix[c].items(), key=lambda kv: kv[1], reverse=True)
        if rank <= len(ranked):
            m, a = ranked[rank-1]
            row.append(f"{names_by_method[m]}: {a:.1f}")
        else:
            row.append("")
    cell_text.append(row)

df_csv = pd.DataFrame(cell_text, index=[f"#{r}" for r in rank_rows], columns=cols)
df_csv.to_csv("plotting_scripts/plots/method_ranks_by_context.csv", index=True)

fig, ax = plt.subplots(figsize=(max(12, 0.8*len(context_range)), 0.6*len(methods)+2))
ax.axis("off")
tbl = ax.table(cellText=cell_text, rowLabels=[f"#{r}" for r in rank_rows], colLabels=cols, loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.2)
plt.tight_layout()
plt.savefig("plotting_scripts/plots/method_ranks_by_context.png", dpi=600)
