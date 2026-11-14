import numpy as np

def generate_trial_data(n_trials=200, trial_duration=0.4, dt=0.1, seed=42):
    """Generates structured trial inputs with varying signal ambiguity."""
    np.random.seed(seed)
    n_steps = int(trial_duration / (dt / 1000.0))
    
    true_labels = np.random.choice([0, 1], size=n_trials)
    difficulties = np.random.uniform(0.1, 1.0, size=n_trials) # signal strength
    
    trials = []
    for label, diff in zip(true_labels, difficulties):
        base = 0.6 if label == 1 else -0.6
        signal = base * diff * np.ones(n_steps)
        # Add trial-specific internal noise
        noise = np.random.normal(0, 0.3, size=n_steps)
        trials.append(signal + noise)
        
    return true_labels, difficulties, np.array(trials), dt