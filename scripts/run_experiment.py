import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import run, second, ms

from src.data import generate_trial_data
from src.reservoir import build_confidence_reservoir
from src.analysis import compute_neuron_confidence_scores

def main():
    print("Initializing Project 7: Candidate Confidence Neurons Analysis...")
    
    n_trials = 150
    trial_duration = 0.3
    dt = 0.1
    n_neurons = 200
    
    true_labels, difficulties, trials, _ = generate_trial_data(n_trials=n_trials, trial_duration=trial_duration, dt=dt)
    
    # Build and simulate reservoir
    neurons, synapses_rec, spike_monitor = build_confidence_reservoir(n_neurons=n_neurons, dt=dt)
    
    total_duration = len(trials.flatten()) * (dt / 1000.0)
    print(f"Running Brian2 simulation for {total_duration:.2f} seconds...")
    run(total_duration * second)
    
    # Extract firing rate matrix per trial
    spike_times = spike_monitor.t / ms
    spike_indices = spike_monitor.i
    steps_per_trial = int(trial_duration / (dt / 1000.0))
    
    X = np.zeros((n_trials, n_neurons))
    for i in range(n_trials):
        t_start = i * (steps_per_trial * dt)
        t_end = (i + 1) * (steps_per_trial * dt)
        mask = (spike_times >= t_start) & (spike_times < t_end)
        if np.sum(mask) > 0:
            counts, _ = np.histogram(spike_indices[mask], bins=np.arange(n_neurons + 1))
            X[i] = counts / (steps_per_trial * dt / 1000.0)
            
    scores, correctness, predictions = compute_neuron_confidence_scores(X, true_labels)
    
    # Identify top confidence neurons
    top_neuron = np.argmax(np.abs(scores))
    print(f"Analysis complete. Top candidate confidence neuron ID: {top_neuron} (Score: {scores[top_neuron]:.4f})")
    
    # Plot & Save
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    plt.bar(range(n_neurons), scores, color='tab:purple', alpha=0.8, width=1.0)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel("Neuron Index", fontsize=11)
    plt.ylabel("Choice-Correctness Correlation ($r$)", fontsize=11)
    plt.title("Project 7: Identifying Candidate Confidence Neurons in Spiking Reservoir", fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/main_result.png", dpi=300)
    print("Saved confidence neuron spectrum plot to figures/main_result.png")

if __name__ == "__main__":
    main()