import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def printResults(accuracies, num_iterations, title):
    bin_labels = sorted(set(accuracies))
    bins = pd.cut(accuracies, bins=len(bin_labels), labels=bin_labels, include_lowest=True)

    bin_counts = bins.value_counts().sort_index()
    bin_percentages = (bin_counts / num_iterations) * 100

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(bin_labels))

    plt.bar(index, bin_percentages.values, bar_width, label=title, color='b', edgecolor='black')

    plt.xlabel('Error rate')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Distribution of error rate over 1000 iterations')
    plt.xticks(index, bin_labels)
    plt.legend()

    plt.tight_layout()
    plt.show()
