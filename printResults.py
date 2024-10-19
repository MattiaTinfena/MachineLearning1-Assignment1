import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def printResults(accuracies, accuraciesLaplace, num_iterations):

    # Usa pandas.cut per suddividere le accuratezze in bin con etichette discrete
    bin_labels = sorted(set(accuracies))  # Etichette dei bin che vuoi usare
    bins = pd.cut(accuracies, bins=len(bin_labels), labels=bin_labels, include_lowest=True)
    binsLaplace = pd.cut(accuraciesLaplace, bins=len(bin_labels), labels=bin_labels, include_lowest=True)

    # Conta quante accuratezze rientrano in ciascun bin
    bin_counts = bins.value_counts().sort_index()
    bin_countsLaplace = binsLaplace.value_counts().sort_index()

    # Converti i conteggi in percentuale
    bin_percentages = (bin_counts / num_iterations) * 100
    bin_percentagesLaplace = (bin_countsLaplace / num_iterations) * 100

    # Plotting del grafico a barre sovrapposto
    plt.figure(figsize=(10, 6))

    bar_width = 0.35  # Larghezza delle barre
    index = np.arange(len(bin_labels))  # Indici per l'asse x

    # Grafico delle accuratezze senza Laplace
    plt.bar(index, bin_percentages.values, bar_width, label='Naive Bayes Classifier', color='b', edgecolor='black')

    # Grafico delle accuratezze con Laplace
    plt.bar(index + bar_width, bin_percentagesLaplace.values, bar_width, label='Laplace smoothing', color='r', edgecolor='black')

    # Configurazione dell'asse x
    plt.xlabel('Error rate')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Distribution of error rate over 1000 iterations')
    plt.xticks(index + bar_width / 2, bin_labels)  # Posiziona le etichette al centro delle barre
    plt.legend()

    # Mostra il grafico
    plt.tight_layout()
    plt.show()