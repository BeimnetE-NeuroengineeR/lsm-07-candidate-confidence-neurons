from brian2 import start_scope, defaultclock, ms, mV, NeuronGroup, Synapses, SpikeMonitor, rand
import numpy as np

def build_confidence_reservoir(n_neurons=200, dt=0.1):
    """Constructs the LIF spiking reservoir for internal state tracking."""
    start_scope()
    defaultclock.dt = dt * ms

    eqs = '''
    dv/dt = (-(v - (-65*mV)) + I_syn + I_ext) / (20*ms) : volt (unless refractory)
    dI_syn/dt = -I_syn / (5*ms) : volt
    dI_ext/dt = -I_ext / (2*ms) : volt
    '''

    neurons = NeuronGroup(n_neurons, eqs, threshold='v > -50*mV', reset='v = -70*mV', refractory=5*ms, method='exact')
    neurons.v = -65*mV + (-50*mV - (-65*mV)) * rand(n_neurons)

    synapses_rec = Synapses(neurons, neurons, model='w : 1', on_pre='I_syn += w * mV')
    synapses_rec.connect(p=0.1)
    
    n_exc = int(0.8 * n_neurons)
    sources = synapses_rec.i[:]
    weights = np.where(sources < n_exc, np.random.rand(len(sources)) * 0.8, -np.random.rand(len(sources)) * 1.5)
    synapses_rec.w = weights

    spike_monitor = SpikeMonitor(neurons)
    return neurons, synapses_rec, spike_monitor