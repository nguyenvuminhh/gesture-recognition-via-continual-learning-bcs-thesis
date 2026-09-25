import os
import re
import numpy as np
import itertools
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.constants import ContinualLearningMethodClass
import json
from plotting_scripts.result_dto import TrainingResultDTO

import matplotlib.pyplot as plt

def draw_accuracy_plots(train_accuracies, context_val_accuracies, val_accuracies, 
                       nof_context, nof_epoch_per_context, log_dir):
    """
    Draw accuracy plots with 3 subplots stacked vertically.
    
    Args:
        train_accuracies: List of training accuracies
        context_val_accuracies: List of context validation accuracies
        val_accuracies: List of overall validation accuracies
        nof_context: Number of contexts
        nof_epoch_per_context: Number of epochs per context
        log_dir: Directory to save the plot
    """
    # Create subplots with 3 plots stacked vertically
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Train Accuracy
    axs[0].plot(range(1, len(train_accuracies) + 1), train_accuracies, color="red", linewidth=2)
    axs[0].set_ylabel("Train Accuracy")
    axs[0].set_title("Training Accuracy")
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Context Validation Accuracy
    axs[1].plot(range(1, len(context_val_accuracies) + 1), context_val_accuracies, color="blue", linewidth=2)
    axs[1].set_ylabel("Context Val Accuracy")
    axs[1].set_title("Context Validation Accuracy")
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Overall Validation Accuracy (So Far)
    axs[2].plot(range(1, len(val_accuracies) + 1), val_accuracies, color="green", linewidth=2)
    axs[2].set_ylabel("SoFar Val Accuracy")
    axs[2].set_title("SoFar Validation Accuracy")
    axs[2].set_xlabel("Epoch")
    axs[2].grid(True, alpha=0.3)
    
    # Add vertical lines at each context boundary for all subplots
    for ax in axs:
        for i in range(1, nof_context):
            context_boundary = i * nof_epoch_per_context+1
            ax.axvline(x=context_boundary, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    filename = os.path.join(log_dir, "val_accuracy_curves_stacked.png")
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    plt.show()

def draw_val_loss_plot(val_losses, log_dir):
    """
    Draw validation loss plot.
    
    Args:
        val_losses: List of validation losses
        log_dir: Directory to save the plot
    """
    plt.figure()
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss vs Epochs')
    filename = os.path.join(log_dir, "val_loss_curve.png")
    plt.savefig(filename, dpi=800)
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, log_dir="./"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    filename = os.path.join(log_dir, "confusion_matrix.png")
    plt.savefig(filename, dpi=800)
    plt.show()

    
# if __name__ == "__main__":
#     # Example usage
#     # Reload the file after environment reset
#     file_path = "log/classification/full_ewc/logs/pointnet2-bi-gru.txt"
#     with open(file_path, "r", encoding="utf-8") as f:
#         new_log_data = f.read()

#     # Extract using regular expressions
#     train_acc_pattern = re.compile(r"Train Instance Accuracy: ([0-9.]+)")
#     context_val_acc_pattern = re.compile(r"\[Context\] Val Instance Accuracy: ([0-9.]+)")
#     sofart_val_acc_pattern = re.compile(r"\[So Far\] Val Instance Accuracy: ([0-9.]+)")

#     train_accuracies = [float(m) for m in train_acc_pattern.findall(new_log_data)]
#     context_val_accuracies = [float(m) for m in context_val_acc_pattern.findall(new_log_data)]
#     val_accuracies = [float(m) for m in sofart_val_acc_pattern.findall(new_log_data)]
    
#     nof_context = 15
#     nof_epoch_per_context = len(train_accuracies) // nof_context if train_accuracies else 5
#     log_dir = "./"
    
#     if train_accuracies and context_val_accuracies and val_accuracies:
#         draw_accuracy_plots(train_accuracies, context_val_accuracies, val_accuracies,
#                             nof_context, nof_epoch_per_context, log_dir)
#         print("Plot saved successfully.")
#     else:
#         print("No data found in the log file.")
if __name__ == "__main__":
    file_name = "training_result.json"
    based_file_path = "log/classification/final_"

    methods = [
        ContinualLearningMethodClass.NONE.value,
        ContinualLearningMethodClass.EWC.value,
        ContinualLearningMethodClass.SI.value,
        ContinualLearningMethodClass.LwF.value,
        ContinualLearningMethodClass.ER.value,
        ContinualLearningMethodClass.AGEM.value
    ]

    method_names = methods.copy()
    method_names[5] = "A-GEM"

    results = {}

    for method in methods:
        file_path = os.path.join(based_file_path + method.lower(), file_name)
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            new_log_data = json.load(f)

        results[method] = TrainingResultDTO(**new_log_data)

    joint_224_result = {
        "context_f1s": 
        0.9879784550224509,
        "so_far_f1s": 
        0.9874958801395642,
        "context_accuracies": 
        0.9874765184721352,
        "so_far_accuracies": 
        0.986850344395742,
    }
    joint_14_result = {
        "context_f1s": 
        0.8990046055701095,
        "so_far_f1s": 
        0.8977939637066448,
        "context_accuracies": 
        0.9173450219160927,
        "so_far_accuracies": 
        0.9173450219160927,
    }
    print("Loaded results for methods:", list(results.keys()))

    height = 7
    width = 9

    if results:
        display_ticks = np.arange(0, 101, 10)  # percentage ticks
        context_range = list(range(1, next(iter(results.values())).nof_context + 1))

        # ===== Plot 1: Context Accuracy + Relative to None =====
        none_context_acc = np.array(results[ContinualLearningMethodClass.NONE.value].context_accuracies) * 100

        fig_context, (ax_context, ax_context_rel) = plt.subplots(2, 1, figsize=(width, height), sharex=False)

        # Main plot
        for method, result in results.items():
            ax_context.plot(context_range, np.array(result.context_accuracies) * 100,
                            label=method_names[methods.index(method)], linewidth=1)
        for c in context_range:
            ax_context.axvline(x=c, color='gray', linestyle='--', linewidth=0.5)
        ax_context.set_ylabel("CCA (%)")
        ax_context.set_yticks(np.arange(0, 101, 5))
        ax_context.set_yticklabels(['' if tick not in display_ticks else f'{tick}'
                                    for tick in np.arange(0, 101, 5)])
        ax_context.axhline(y=joint_224_result["context_accuracies"] * 100,
                        color='red', linestyle='-.', linewidth=1, label="JOINT")
        ax_context.legend()
        ax_context.set_xlabel("Context")

        # Relative subplot
        for method, result in results.items():
            rel_acc = (np.array(result.context_accuracies) * 100) - none_context_acc
            ax_context_rel.plot(context_range, rel_acc, label=method_names[methods.index(method)], linewidth=1)
        for c in context_range:
            ax_context_rel.axvline(x=c, color='gray', linestyle='--', linewidth=0.5)
        ax_context_rel.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax_context_rel.set_ylabel("CCA-diff (%)")
        ax_context_rel.set_xlabel("Context")
        ax_context_rel.legend()
        plt.tight_layout()
        fig_context.savefig("plotting_scripts/plots/cca.png", dpi=800)

        # ===== Plot 2: So-far Accuracy + Relative to None =====
        none_sofar_acc = np.array(results[ContinualLearningMethodClass.NONE.value].so_far_accuracies) * 100

        fig_sofar, (ax_sofar, ax_sofar_rel) = plt.subplots(2, 1, figsize=(width, height), sharex=False)

        # Main plot
        for method, result in results.items():
            ax_sofar.plot(context_range, np.array(result.so_far_accuracies) * 100,
                        label=method_names[methods.index(method)], linewidth=1)
        for c in context_range:
            ax_sofar.axvline(x=c, color='gray', linestyle='--', linewidth=0.5)
        ax_sofar.set_ylabel("SCA (%)")
        ax_sofar.set_yticks(np.arange(0, 101, 5))
        ax_sofar.set_yticklabels(['' if tick not in display_ticks else f'{tick}'
                                for tick in np.arange(0, 101, 5)])
        ax_sofar.axhline(y=joint_224_result["so_far_accuracies"] * 100,
                        color='red', linestyle='-.', linewidth=1, label="JOINT")
        ax_sofar.legend()
        ax_sofar.set_xlabel("Context")

        # Relative subplot
        for method, result in results.items():
            rel_acc = (np.array(result.so_far_accuracies) * 100) - none_sofar_acc
            ax_sofar_rel.plot(context_range, rel_acc, label=method_names[methods.index(method)], linewidth=1)
        for c in context_range:
            ax_sofar_rel.axvline(x=c, color='gray', linestyle='--', linewidth=0.5)
        ax_sofar_rel.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax_sofar_rel.set_ylabel("SCA-diff (%)")
        ax_sofar_rel.set_xlabel("Context")
        ax_sofar_rel.legend()

        plt.tight_layout()

        fig_sofar.savefig("plotting_scripts/plots/sca.png", dpi=800)

        # ===== Plot 3: Train Accuracy for Each Method =====
        fig_train, ax_train = plt.subplots(figsize=(10, height))
        for method, result in results.items():
            if method != ContinualLearningMethodClass.ER.value:
                ax_train.plot(range(1, len(result.train_accuracies) + 1),
                    np.array(result.train_accuracies) * 100,
                    label=method_names[methods.index(method)], linewidth=1)
        ax_train.set_ylabel("Training Accuracy (%)")
        ax_train.set_xlabel("Epoch")
        ax_train.set_yticks(np.arange(0, 101, 5))
        ax_train.set_yticklabels(['' if tick not in display_ticks else f'{tick}'
                      for tick in np.arange(0, 101, 5)])
        ax_train.legend()
        plt.tight_layout()
        fig_train.savefig("plotting_scripts/plots/train_accuracy_per_epoch.png", dpi=800)


        # ===== Five subplots per metric: each shows one compared method vs baselines (None, JOINT)
        comp_methods = [
            ContinualLearningMethodClass.EWC.value,
            ContinualLearningMethodClass.SI.value,
            ContinualLearningMethodClass.LwF.value,
            ContinualLearningMethodClass.ER.value,
            ContinualLearningMethodClass.AGEM.value,
        ]

        def plot_five(metric_attr, ylabel, joint_key, outfile):
            none_series = np.array(getattr(results[ContinualLearningMethodClass.NONE.value], metric_attr)) * 100
            joint_line = joint_224_result[joint_key] * 100
            fig, axes = plt.subplots(5,1, figsize=(8, 11), sharey=True)
            for i, m in enumerate(comp_methods):
                r = results[m]
                y = np.array(getattr(r, metric_attr)) * 100
                axes[i].plot(context_range, none_series, linewidth=1, label="None")
                axes[i].plot(context_range, y, linewidth=1, label=method_names[methods.index(m)])
                axes[i].axhline(y=joint_line, linestyle='-.', linewidth=1, label="JOINT")
                for c in context_range:
                    axes[i].axvline(x=c, color='gray', linestyle='--', linewidth=0.5)
                axes[i].set_title(method_names[methods.index(m)])
                axes[i].set_xlabel("Context")
                axes[i].set_yticks(np.arange(0, 101, 20))
                axes[i].set_yticklabels(['' if tick not in display_ticks else f'{tick}' for tick in np.arange(0, 101, 20)])
                if i == 0:
                    axes[i].set_ylabel(ylabel)
                axes[i].legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(outfile, dpi=800)

        plot_five("context_accuracies", "CCA (%)", "context_accuracies", "plotting_scripts/plots/ccaextended.png")
        plot_five("so_far_accuracies", "SCA (%)", "so_far_accuracies", "plotting_scripts/plots/scaextended.png")