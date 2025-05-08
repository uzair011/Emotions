import numpy as np
import matplotlib.pyplot as plt


def plot_latency_over_time():
    np.random.seed(42)

    frames = np.arange(1, 101)
    face_detection = np.random.normal(33, 5, 100)
    emotion_pred = np.random.normal(50, 8, 100)
    total_latency = face_detection + emotion_pred

    plt.figure(figsize=(10, 5))
    plt.plot(frames, total_latency, label='Total Latency', color='purple', alpha=0.7)
    plt.axhline(y=83, color='r', linestyle='--', label='Average (83ms)')
    plt.fill_between(frames, total_latency - 10, total_latency + 10, color='purple', alpha=0.1)
    plt.title('Real-Time Latency Over 100 Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('latency_plot.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_latency_over_time()
