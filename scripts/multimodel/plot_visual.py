import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def plot_multimodal_fusion():


    data = {
        'Model': ['Visual (ResNet-18)', 'Audio (1D-CNN)', 'Fusion (Weighted)'],
        'Accuracy': [68, 72, 75],
        'Conflict Rate': [None, None, 18]
    }
    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(df['Model'], df['Accuracy'], color=['blue', 'red', 'purple'], alpha=0.7)
    ax1.set_ylabel('Accuracy (%)', color='black')
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(df['Model'], df['Conflict Rate'], marker='o', color='green', label='Conflict Rate')
    ax2.set_ylabel('Conflict Rate (%)', color='green')
    ax2.set_ylim(0, 30)

    plt.title('Multimodal Fusion Performance')
    fig.tight_layout()
    plt.savefig('fusion_performance.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_multimodal_fusion()
