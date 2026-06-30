import numpy as np
import bayesflow as bf
import matplotlib.pyplot as plt
import tensorflow as tf
# %%

def LIF_model(tau_m, R, amp, noise_std, V_rest=-65.0, V_th=-50.0, V_reset=-65.0):
    n_steps = 1000
    dt = 1.0  # ms

    V = V_rest * np.ones(n_steps)
    I = np.zeros(n_steps)
    spikes = np.zeros(n_steps)

    # Rectangular pulse input + noise
    I[300:700] = amp
    I += noise_std * np.random.randn(n_steps)

    for t in range(1, n_steps):
        dV = dt * (-(V[t-1] - V_rest) + R * I[t]) / tau_m
        V[t] = V[t-1] + dV
        if V[t] >= V_th:
            V[t] = V_reset
            spikes[t] = 1  # mark spike

    return I, V, spikes

# %%

def prior():
    # Specific capacitance: ~10 nF/mm², tightly distributed
    c_m = np.random.lognormal(mean=np.log(10), sigma=0.05)  # nF/mm²

    # Surface area: 0.01–0.1 mm²
    A = np.random.uniform(0.01, 0.1)  # mm²

    # Specific resistance: log-normal around 1 MΩ·mm²
    r_m = np.random.lognormal(mean=np.log(1.0), sigma=0.5)  # MΩ·mm²

    # Derived values 
    tau_m = r_m * c_m          # ms (MΩ·nF = ms)
    R = r_m / A                # MΩ

    return dict(tau_m=tau_m, R=R)
# %%

def likelihood(tau_m, R):
    tau_m = tau_m
    R = R
    
    # Input current with noisy
    amp = np.random.uniform(0.05, 4)
    noise = np.random.uniform(0.005, 0.1)

    # Generate data
    I, V, spikes = LIF_model(tau_m, R, amp, noise)
    return dict(I=I, V=V, spikes=spikes)
# %%

def show_sample_observation(sample, index, V_reset=-65.0):
    
    I = sample["I"][index]
    V = sample["V"][index]
    spikes = sample["spikes"][index]
    time = np.linspace(0, 1.0, len(I))

    plt.figure(figsize=(12, 5))

    # Input current
    plt.subplot(2, 1, 1)
    plt.plot(time, I, color='tab:blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Current (nA)")
    plt.title("Input Current")

    # Membrane potential
    plt.subplot(2, 1, 2)
    plt.plot(time, V, label="Membrane potential", color='tab:blue')
    # Horizontal threshold line
    plt.axhline(-50, color='red', linestyle='--', linewidth=1, label='Threshold')
    # Draw vertical lines for spikes
    spike_times = time[np.where(spikes)[0]]
    for t in spike_times:
        plt.vlines(t, ymin=V_reset, ymax=-45.0, color='tab:blue', linewidth=1)

    plt.ylim(-70, -40)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Membrane Potential with Spike Lines")
    plt.tight_layout()
    plt.show()
# %%

simulator = bf.make_simulator([prior, likelihood])
training_sample = simulator.sample(1000)
validation_sample = simulator.sample(1000)
# %%

show_sample_observation(training_sample, index=4)
show_sample_observation(training_sample, index=8)
show_sample_observation(validation_sample, index=12)
show_sample_observation(validation_sample, index=8)

# %%

adapter = (
    bf.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(["tau_m", "R"], into="inference_variables")
    .as_time_series(["I", "V", "spikes"])
    .concatenate(["V", "I", "spikes"], into="summary_variables")
)
print(adapter)
ada_data = adapter(training_sample)
print(ada_data["summary_variables"].shape)
# %%

summary_network = bf.networks.TimeSeriesNetwork()
inference_network = bf.networks.CouplingFlow()
workflow = bf.BasicWorkflow(
    inference_network=inference_network,
    summary_network=summary_network,
    adapter=adapter,
    simulator=simulator
)

history = workflow.fit_online(epochs = 15)

from keras import backend as K

# After training:
K.clear_session()
# %%

from pathlib import Path

filepath = Path("checkpoints") / "my_lif_model.keras"
filepath.parent.mkdir(parents=True, exist_ok=True)
workflow.approximator.save(filepath=filepath)
print(f"Model saved at: {filepath.resolve()}")
# %%
loaded_approximator = tf.keras.models.load_model("checkpoints/my_lif_model.keras")

# %%

# Set the number of posterior draws you want to get
num_samples = 100

# Simulate validation data (unseen during training)
val_sims = simulator.sample(2000)

# Obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = loaded_approximator.sample(conditions=val_sims, num_samples=num_samples)
post_draws.keys()
print(post_draws["tau_m"].shape)
# %%

bf.diagnostics.plots.calibration_histogram(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=['tau_m', 'R']
)

bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=val_sims,
    dataset_id=0,
    variable_names=['tau_m', 'R'],
)

bf.diagnostics.plots.recovery(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=['tau_m', 'R']
)

bf.diagnostics.plots.z_score_contraction(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=['tau_m', 'R']
)
# %%

bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=val_sims,
    difference=True,
    variable_names=['tau_m', 'R']
)


