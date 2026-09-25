import json
import matplotlib.pyplot as plt

with open("log/classification/agem_for_dot_prod/training_result.json", "r") as f:
    data = json.load(f)

dot_products = data.get("dot_products", {})

def extract_metric(items, key):
    out = []
    for it in items:
        if isinstance(it, dict):
            v = it.get(key, None)
        else:
            v = getattr(it, key, None)
        if v is not None:
            out.append(v)
    return out

def plot_metric(metric_name, ylabel):
    plt.figure(figsize=(12, 6))
    current_x = 0
    for k in sorted(dot_products.keys(), key=lambda x: int(x)):
        items = dot_products[k]
        if not isinstance(items, list) or len(items) == 0:
            continue
        vals = extract_metric(items, metric_name)

        # Print average for each key
        avg_val = sum(vals) / len(vals) if vals else 0.0
        print(f"{metric_name} - Key {k}: average = {avg_val:.4f}")

        x_vals = list(range(current_x, current_x + len(vals)))
        plt.plot(x_vals, vals, label=f"Key {k}")
        plt.axvline(x=current_x, linestyle='--', linewidth=1)
        current_x += len(vals)

    plt.xlabel("Index (continuous)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} (Continuous Plot)")
    plt.legend()
    plt.grid(True)

# Cosine similarity stats
for k in sorted(dot_products.keys(), key=lambda x: int(x)):
    items = dot_products[k]
    if not isinstance(items, list) or len(items) == 0:
        continue
    cos_vals = extract_metric(items, "cos_sim")
    total = len(cos_vals)
    pos = sum(1 for v in cos_vals if v > 0)
    neg = sum(1 for v in cos_vals if v < 0)
    avg = sum(cos_vals) / total if total else 0.0
    avg_neg = (sum(v for v in cos_vals if v < 0) / total) if total else 0.0
    print(f"Cosine Similarity - Key {k}: {pos/total*100:.2f}% positive, "
          f"{neg/total*100:.2f}% negative, average {avg:.4f}, avg negative {avg_neg:.4f}")

# Plot metrics with averages printed
plot_metric("cos_sim", "Cosine Similarity")
plot_metric("magnitude_ratio", "Magnitude Ratio")
plot_metric("projection_loss", "Projection Loss")

plt.show()
