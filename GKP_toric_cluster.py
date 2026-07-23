import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, eye, hstack, vstack, kron
from scipy.special import ndtr
import pymatching
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import json
import os
import time

sqrt_pi = np.sqrt(np.pi) # to write less later on
sqrt_2 = np.sqrt(2)

n_trials = 5000


def gkp_physical_error_rate(sigma, number_of_intervals=100):
    """Probability that ideal GKP correction produces a Pauli fault."""
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")

    probability = np.zeros_like(sigma)
    for m in range(number_of_intervals):
        lower = (2 * m + 0.5) * sqrt_pi / sigma
        upper = (2 * m + 1.5) * sqrt_pi / sigma
        probability += 2 * (ndtr(-lower) - ndtr(-upper))
    return float(probability) if probability.ndim == 0 else probability


def decimal_probability_tick(value, _position):
    if value <= 0:
        return ""
    precision = 3 if value >= 0.01 else 4
    return f"{value:.{precision}f}".rstrip("0").rstrip(".")


def format_log_probability_axis(axis):
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=30))
    axis.set_major_formatter(FuncFormatter(decimal_probability_tick))
    axis.set_minor_locator(
        LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9), numticks=100)
    )
    axis.set_minor_formatter(NullFormatter())

def toric_rep_code(n):
    row = np.repeat(np.arange(n), 2)
    col = np.zeros(2 * n, dtype=int)
    for i in range(n):
        col[2 * i] = i
        col[2 * i + 1] = (i + 1) % n
    data = np.ones(2 * n, dtype=np.uint8)
    return csc_matrix((data, (row, col)), shape=(n, n))

def toric_x_stab(d):
    Hr = toric_rep_code(d)
    I = eye(d, dtype=np.uint8)
    H = hstack([kron(Hr, I), kron(I, Hr)])
    H.data = H.data % 2
    H.eliminate_zeros()
    return csc_matrix(H)

def toric_logical_x(d):
    num_qubits = 2 * (d ** 2)
    log1 = np.zeros(num_qubits, dtype=np.uint8)
    log2 = np.zeros(num_qubits, dtype=np.uint8)
    
    for i in range(d):
        log1[i] = 1
        log2[d**2 + i*d] = 1 
        
    logicals = np.vstack([log1, log2])
    return csc_matrix(logicals)


#more rounds of measurement till no noise fix 

def run_gkp_toric_spacetime_simulation(d, sigma_vals, max_trials=50000, min_errors=100, min_trials=1000, save_filename="simulation_checkpoint.json"):
    Hx = toric_x_stab(d)
    logical_x = toric_logical_x(d)
    num_qubits = Hx.shape[1]
    num_stabilisers = Hx.shape[0]
    T = d 
    
    H_space = vstack([kron(eye(T, dtype=np.uint8), Hx), csc_matrix((num_stabilisers, T * num_qubits), dtype=np.uint8)])
    I_stab = eye(num_stabilisers, dtype=np.uint8)
    
    Row_time = []
    for i in range(T + 1):
        row_blocks = []
        for j in range(T):
            if i == j or i == j + 1:
                row_blocks.append(I_stab)
            else:
                row_blocks.append(csc_matrix((num_stabilisers, num_stabilisers), dtype=np.uint8))
        Row_time.append(hstack(row_blocks))
    H_time = vstack(Row_time)
    
    H_3D = hstack([H_space, H_time])
    H_3D.data = H_3D.data % 2
    H_3D.eliminate_zeros()

    #data
    if os.path.exists(save_filename):
        with open(save_filename, 'r') as f:
            saved_data = json.load(f)
        print(f"Loaded existing checkpoint data from {save_filename}")
    else:
        saved_data = {}

    str_d = str(d)
    if str_d not in saved_data:
        saved_data[str_d] = {"sigma": [], "rate": [], "error_bar": [], "total_trials": [], "total_errors": []}

    last_save_time = time.time()

    for sigma in sigma_vals:
        sigma_rounded = round(float(sigma), 5)
        
        #get data from file for quicker plots
        if sigma_rounded in saved_data[str_d]["sigma"]:
            idx = saved_data[str_d]["sigma"].index(sigma_rounded)
            existing_trials = saved_data[str_d]["total_trials"][idx]
            existing_errors = saved_data[str_d]["total_errors"][idx]
            
            if existing_trials >= max_trials or (existing_errors >= min_errors and existing_trials >= min_trials):
                print(f"Existing data for d={d}, sigma={sigma_rounded} (has {existing_trials} trials)")
                continue
            else:
                # if interupted bc laptop died a few times 
                print(f"Resuming d={d}, sigma={sigma_rounded} from {existing_trials} to max {max_trials} trials...")
                logical_errors = existing_errors
                trials_run = existing_trials
        else:
            # new data if none saved
            print(f"Simulating new data: d={d}, sigma={sigma_rounded}...")
            logical_errors = 0
            trials_run = 0

        while trials_run < max_trials:
            batch_size = min(500, max_trials - trials_run)  #batch for daster processing still takes a while tho
            
            for _ in range(batch_size):
                qubit_errors_per_step = []
                qubit_weights_per_step = []
                cumulative_ep = np.zeros(num_qubits, dtype=np.uint8)
                
                for t in range(T):
                    dp = np.random.normal(0, sigma, num_qubits)
                    sp_vals = ((dp + sqrt_pi / 2) % sqrt_pi) - sqrt_pi / 2
                    ep_slice = (np.round((dp - sp_vals) / sqrt_pi).astype(int)) % 2
                    cumulative_ep = (cumulative_ep + ep_slice) % 2
                    qubit_errors_per_step.append(ep_slice)
                    space_w = ((sqrt_pi - np.abs(sp_vals)) ** 2 - sp_vals ** 2) / (2 * sigma ** 2)
                    qubit_weights_per_step.append(space_w)
                
                ideal_syndromes = []
                current_qubit_state = np.zeros(num_qubits, dtype=np.uint8)
                for t in range(T):
                    current_qubit_state = (current_qubit_state + qubit_errors_per_step[t]) % 2
                    ideal_syndromes.append((Hx @ current_qubit_state) % 2)
                    
                meas_weights_per_step = []
                meas_errors_per_step = []
                noisy_syndromes = []
                for t in range(T):
                    meas_dp = np.random.normal(0, sigma, num_stabilisers)
                    meas_sp = ((meas_dp + sqrt_pi / 2) % sqrt_pi) - sqrt_pi / 2
                    meas_ep = (np.round((meas_dp - meas_sp) / sqrt_pi).astype(int)) % 2
                    noisy_slice = (ideal_syndromes[t] + meas_ep) % 2
                    noisy_syndromes.append(noisy_slice)
                    meas_errors_per_step.append(meas_ep)
                    time_w = ((sqrt_pi - np.abs(meas_sp)) ** 2 - meas_sp ** 2) / (2 * sigma ** 2)
                    meas_weights_per_step.append(time_w)
                    #final one read our data qubit 
                    
                detectors = []
                prev_syndrome = np.zeros(num_stabilisers, dtype=np.uint8) 
                for t in range(T + 1):
                    current_syn = noisy_syndromes[t] if t < T else (Hx @ cumulative_ep) % 2
                    dt = (current_syn + prev_syndrome) % 2
                    detectors.append(dt)
                    prev_syndrome = current_syn
                    
                full_detector_syndrome = np.concatenate(detectors)
                full_weights_vector = np.concatenate(qubit_weights_per_step + meas_weights_per_step)
                assert H_3D.shape[1] == len(full_weights_vector)
                
                # PyMatching weights belong to graph edges and must be supplied
                # when the Matching object is constructed. Passing weights to
                # decode() is ignored by PyMatching 2.x.
                shot_matching = pymatching.Matching.from_check_matrix(
                    H_3D, weights=full_weights_vector
                )
                correction = shot_matching.decode(full_detector_syndrome)
                cumulative_correction = np.zeros(num_qubits, dtype=np.uint8)
                for t in range(T):
                    layer_start = t * num_qubits
                    cumulative_correction = (cumulative_correction + correction[layer_start:layer_start + num_qubits]) % 2
                    
                residual_error = (cumulative_ep + cumulative_correction) % 2
                if np.any((logical_x @ residual_error) % 2):
                    logical_errors += 1

            trials_run += batch_size
            
            # stop if enough errors 
            if logical_errors >= min_errors and trials_run >= min_trials:
                break

        # rates
        rate = logical_errors / trials_run
        err = np.sqrt(rate * (1 - rate) / trials_run) if trials_run > 0 else 0

        # update data in correct columns 
        if sigma_rounded in saved_data[str_d]["sigma"]:
            idx = saved_data[str_d]["sigma"].index(sigma_rounded)
            saved_data[str_d]["rate"][idx] = float(rate)
            saved_data[str_d]["error_bar"][idx] = float(err)
            saved_data[str_d]["total_trials"][idx] = int(trials_run)
            saved_data[str_d]["total_errors"][idx] = int(logical_errors)
        else:
            saved_data[str_d]["sigma"].append(sigma_rounded)
            saved_data[str_d]["rate"].append(float(rate))
            saved_data[str_d]["error_bar"].append(float(err))
            saved_data[str_d]["total_trials"].append(int(trials_run))
            saved_data[str_d]["total_errors"].append(int(logical_errors))

        #5 mins = save data
        current_time = time.time()
        if (current_time - last_save_time) > 300 or sigma == sigma_vals[-1]:
            with open(save_filename, 'w') as f:
                json.dump(saved_data, f, indent=4)
            print(f"--> Saved progress checkpoint to {save_filename} (Total Trials: {trials_run}, Total Errors: {logical_errors})")
            last_save_time = current_time

    # sort data
    sorted_indices = np.argsort(saved_data[str_d]["sigma"])
    out_sigmas = [saved_data[str_d]["sigma"][i] for i in sorted_indices]
    out_rates = [saved_data[str_d]["rate"][i] for i in sorted_indices]
    out_errs = [saved_data[str_d]["error_bar"][i] for i in sorted_indices]

    return out_sigmas, out_rates, out_errs

sigma_vals = np.array([
    0.30, 0.35, 0.38,
    0.390, 0.395, 0.400, 0.405, 0.410,
    0.42, 0.45, 0.50,
])

distances = [3, 5, 9, 15, 20]

max_trials = 2_000_000
min_errors = 500
min_trials = 20_000
checkpoint_file = "gkp_toric_spacetime_results_3.json"

results_by_distance = {}

for d in distances:
    max_allowed_shots = 2_000_000 #might need to increase even tho high bc low sigma still has rly high error
    
    rates_sigma, rates, errs = run_gkp_toric_spacetime_simulation(
        d=d, 
        sigma_vals=sigma_vals, 
        max_trials=max_allowed_shots, 
        min_errors= 500, 
        min_trials=20_000,
        save_filename=checkpoint_file
    )
    results_by_distance[d] = (
        np.asarray(rates_sigma, dtype=float),
        np.asarray(rates, dtype=float),
        np.asarray(errs, dtype=float),
    )

# Plot 1: the original logical-error-rate-versus-sigma figure.
fig_sigma, ax_sigma = plt.subplots(figsize=(8, 6))
for d, (rates_sigma, rates, errs) in results_by_distance.items():
    ax_sigma.errorbar(
        rates_sigma, rates, yerr=errs, marker="o", capsize=3,
        label=f"d={d}, T={d}",
    )

ax_sigma.set_yscale("log")
format_log_probability_axis(ax_sigma.yaxis)
ax_sigma.set_xlabel(r"Gaussian displacement spread ($\sigma$)")
ax_sigma.set_ylabel("Logical error rate")
ax_sigma.set_title("GKP-toric logical error rate versus sigma")
ax_sigma.legend()
ax_sigma.grid(True, which="both", ls="--", alpha=0.5)
fig_sigma.tight_layout()
fig_sigma.savefig(
    "GKP_toric_threshold_plot.png",
    dpi=300,
    bbox_inches="tight",
)

# Plot 2: the same logical data with sigma converted to physical GKP error rate.
fig_physical, ax_physical = plt.subplots(figsize=(8, 6))
for d, (rates_sigma, rates, errs) in results_by_distance.items():
    physical_rates = gkp_physical_error_rate(rates_sigma)
    ax_physical.errorbar(
        physical_rates, rates, yerr=errs, marker="o", capsize=3,
        label=f"d={d}, T={d}",
    )

ax_physical.set_xscale("log")
ax_physical.set_yscale("log")
format_log_probability_axis(ax_physical.xaxis)
format_log_probability_axis(ax_physical.yaxis)
ax_physical.set_xlabel(r"Physical GKP error probability ($p_{\mathrm{phys}}$)")
ax_physical.set_ylabel("Logical error rate")
ax_physical.set_title("GKP-toric logical versus physical error rate")
ax_physical.legend()
ax_physical.grid(True, which="both", ls="--", alpha=0.5)
fig_physical.tight_layout()
fig_physical.savefig(
    "GKP_toric_logical_vs_physical_probability.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

#3564679