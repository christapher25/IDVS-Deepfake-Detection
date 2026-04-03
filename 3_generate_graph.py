import matplotlib.pyplot as plt
import pickle
import os

# --- CONFIG ---
HISTORY_PATH = 'models/history.pkl'
OUTPUT_IMAGE = 'training_results_smooth.png'


def smooth_curve(scalars, weight=0.85):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_history():
    if not os.path.exists(HISTORY_PATH):
        print("Error: No history found.")
        return

    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)

    # Smooth the data
    acc = smooth_curve(history['accuracy'])
    val_acc = smooth_curve(history['val_accuracy'])
    loss = smooth_curve(history['loss'])
    val_loss = smooth_curve(history['val_loss'])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Acc', color='green', linestyle='--')
    plt.plot(epochs, val_acc, label='Validation Acc', color='darkgreen', linewidth=2)
    plt.title('Accuracy (Zero to Hero)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', color='red', linestyle='--')
    plt.plot(epochs, val_loss, label='Validation Loss', color='darkred', linewidth=2)
    plt.title('Loss Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"✅ Graph saved: {OUTPUT_IMAGE}")
    plt.show()


if __name__ == "__main__":
    plot_history()