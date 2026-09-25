import json
import matplotlib.pyplot as plt
import numpy as np

from data_utils.constants import ContinualLearningMethodClass


def load_dict(method: str, value: str) -> dict:
    path_prefix = "log/classification/tune_"
    path_suffix = "/training_result.json"
    
    method_str = method.lower()
    value_str = value.lower()

    path = f"{path_prefix}{method_str}_{value_str}{path_suffix}"

    with open(path, "r") as f:
        return json.load(f)
    

methods = [ContinualLearningMethodClass.EWC, ContinualLearningMethodClass.SI, ContinualLearningMethodClass.LwF]
values = ["1e-2", "1e-1", "1", "10", "100"]

labels = ["0.01", "0.1", "1", "10", "100"]

fig, axes = plt.subplots(len(methods), 1, figsize=(8, 11))

for ax, method in zip(axes, methods):
    xs, ys = [], []
    for i, value in enumerate(values):
        result = load_dict(method, value)
        avg_context = np.mean(result["context_accuracies"])
        avg_sofar = np.mean(result["so_far_accuracies"])
        xs.append(avg_context)
        ys.append(avg_sofar)

        ax.plot([avg_context], [avg_sofar], marker="o", linestyle="--", label=labels[i])

    ax.set_title(f"{method}")
    ax.set_xlabel("Average Current-Context Accuracy (CCA)")
    ax.set_ylabel("Average Seen-Context Accuracy (SCA)")
    ax.legend(title="Regularization Strength (λ)")
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("plotting_scripts/plots/tune_plot.png", dpi=800)
