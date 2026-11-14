import numpy as np
from sklearn.linear_model import LogisticRegression

def compute_neuron_confidence_scores(spike_rates, true_labels):
    """Computes choice accuracy per trial and evaluates individual neuron point-biserial correlations."""
    X = spike_rates
    
    # Train population classifier to determine trial correctness
    model = LogisticRegression(C=1.0)
    model.fit(X, true_labels)
    predictions = model.predict(X)
    
    # Correct vs incorrect trials (1 = correct, 0 = error)
    correctness = (predictions == true_labels).astype(int)
    
    # Compute correlation between each neuron's firing rate and trial correctness
    scores = []
    n_neurons = X.shape[1]
    
    for neuron_idx in range(n_neurons):
        rates = X[:, neuron_idx]
        if np.std(rates) == 0:
            scores.append(0.0)
        else:
            corr = np.corrcoef(rates, correctness)[0, 1]
            scores.append(0.0 if np.isnan(corr) else corr)
            
    return np.array(scores), correctness, predictions