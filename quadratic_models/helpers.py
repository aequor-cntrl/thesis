import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHECKPOINT_STEP = 40
NUM_TIMESTEPS = 100
T_CORRUPTION = 50
DETECTION_DELAYS = [0, 4, 16]

def momentum_1d(x_0, alpha, gamma, t_corrupt, detection_delay, delta, strategy, clean_x, clean_velocities, checkpoint_step=CHECKPOINT_STEP):
    x = [x_0]
    gradients = [0]
    velocities = [0]
    for t in range(NUM_TIMESTEPS + detection_delay):
        prev_velocity = velocities[-1]
        if t == t_corrupt + (detection_delay + 1):
            if strategy == "corr_clean":
                prev_velocity = clean_velocities[t]
            elif strategy == "clean_corr":
                x[-1] = clean_x[t]
            elif strategy == "clean_reset" or strategy == "corr_reset":
                prev_velocity = 0
            elif strategy == "rollback":
                if checkpoint_step > t:
                    checkpoint_step = 0
                corrected_x, corrected_velocity = 0, 0
                for rollback_step in range(0, t - checkpoint_step):
                    if rollback_step == 0:
                        corrected_velocity = velocities[checkpoint_step]
                        corrected_x = x[checkpoint_step]
                    corrected_gradient = 2 * corrected_x
                    corrected_velocity = gamma * corrected_velocity + alpha * corrected_gradient
                    corrected_x = corrected_x - corrected_velocity
                x[-1] = corrected_x
                prev_velocity = corrected_velocity
                velocities[-1] = corrected_velocity
        gradient = 2 * x[-1]
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset": 
            gradient += delta
        velocity = gamma * prev_velocity + alpha * gradient
        x.append(x[-1] - velocity)
        gradients.append(gradient)
        velocities.append(velocity)
    return np.array(x), gradients, velocities

def momentum_2d(x_0, y_0, alpha, gamma, t_corrupt, detection_delay, delta, strategy, clean_x, clean_velocities, checkpoint_step=CHECKPOINT_STEP):
    x = [np.array([x_0, y_0])]
    gradients = [np.array([0,0])]
    velocities = [np.array([0,0])]
    for t in range(NUM_TIMESTEPS + detection_delay):
        prev_velocity = velocities[-1]
        if t == t_corrupt + (detection_delay + 1):
            if strategy == "corr_clean":
                prev_velocity = clean_velocities[t]
            elif strategy == "clean_corr":
                x[-1] = clean_x[t]
            elif strategy == "clean_reset" or strategy == "corr_reset":
                prev_velocity = np.array([0,0])
            elif strategy == "rollback":
                if checkpoint_step > t:
                    checkpoint_step = 0
                corrected_x, corrected_velocity = np.array([0,0]), np.array([0,0])
                for rollback_step in range(0, t - checkpoint_step):
                    if rollback_step == 0:
                        corrected_velocity = velocities[checkpoint_step]
                        corrected_x = x[checkpoint_step]
                    corrected_gradient = np.array([2*corrected_x[0] + corrected_x[1], corrected_x[0] + 2*corrected_x[1]])
                    corrected_velocity = gamma * corrected_velocity + alpha * corrected_gradient
                    corrected_x = corrected_x - corrected_velocity
                x[-1] = corrected_x
                prev_velocity = corrected_velocity
                velocities[-1] = corrected_velocity
        gradient = np.array([2*x[-1][0] + x[-1][1], x[-1][0] + 2*x[-1][1]])
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset":
            gradient += np.array([delta, 0])
        velocity = gamma * prev_velocity + alpha * gradient
        x.append(x[-1] - velocity)
        gradients.append(gradient)
        velocities.append(velocity)
    return np.array(x), np.array(gradients), np.array(velocities)

def adam_1d(x_0, alpha, beta_1, beta_2, t_corrupt, detection_delay, delta, strategy, clean_x, clean_m, clean_v, checkpoint_step=CHECKPOINT_STEP):
    x = [x_0]
    m_values = [0]
    v_values = [0]
    timestep = -1
    for t in range(NUM_TIMESTEPS + detection_delay):
        prev_m = m_values[-1]
        prev_v = v_values[-1]
        if strategy != "clean_reset_optim" and strategy != "corr_reset_optim":
            timestep = t
        else:
            timestep += 1
        if t == t_corrupt + (detection_delay + 1):
            if strategy == "clean_clean_corr":
                x[-1] = clean_x[timestep]
                prev_m = clean_m[timestep]
            elif strategy == "clean_corr_clean":
                x[-1] = clean_x[timestep]
                prev_v = clean_v[timestep]
            elif strategy == "clean_corr_corr":
                x[-1] = clean_x[timestep]
            elif strategy == "corr_clean_clean":
                prev_m = clean_m[timestep]
                prev_v = clean_v[timestep]
            elif strategy == "corr_clean_corr":
                prev_m = clean_m[timestep]
            elif strategy == "corr_corr_clean":
                prev_v = clean_v[timestep]
            elif strategy == "rollback":
                if checkpoint_step > t:
                    checkpoint_step = 0
                corrected_x, corrected_m, corrected_v = 0, 0, 0
                for rollback_step in range(0, t - checkpoint_step):
                    if rollback_step == 0:
                        corrected_m = m_values[checkpoint_step]
                        corrected_v = v_values[checkpoint_step]
                        corrected_x = x[checkpoint_step]
                    corrected_gradient = 2 * corrected_x
                    corrected_m = beta_1 * corrected_m + (1-beta_1)*corrected_gradient
                    corrected_v = beta_2 * corrected_v + (1-beta_2)*corrected_gradient**2
                    corrected_hat_m = (corrected_m)/(1-beta_1**(checkpoint_step+rollback_step+1))
                    corrected_hat_v = (corrected_v)/(1-beta_2**(checkpoint_step+rollback_step+1))
                    corrected_x = corrected_x - ((alpha)/(np.sqrt(corrected_hat_v) + 10**(-8)))*corrected_hat_m
                x[-1] = corrected_x
                prev_m = corrected_m
                m_values[-1] = corrected_m
                prev_v = corrected_v
                v_values[-1] = corrected_v
            elif strategy == "clean_reset_optim" or strategy == "corr_reset_optim" or strategy == "clean_reset_moments" or strategy == "corr_reset_moments":
                prev_m = 0
                prev_v = 0
            if strategy == "clean_reset_optim" or strategy == "corr_reset_optim":
                timestep = 0
        gradient = 2 * x[-1]
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset_optim" and strategy != "clean_reset_moments":
            gradient += delta
        m = beta_1 * prev_m + (1-beta_1)*gradient
        v = beta_2 * prev_v + (1-beta_2)*gradient**2
        hat_m = (m)/(1-beta_1**(timestep+1))
        hat_v = (v)/(1-beta_2**(timestep+1))
        x.append(x[-1] - ((alpha)/(math.sqrt(hat_v) + 10**(-8)))*hat_m)
        m_values.append(m)
        v_values.append(v)
    return np.array(x), np.array(m_values), np.array(v_values)

def adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corrupt, detection_delay, delta, strategy, clean_x, clean_m, clean_v, checkpoint_step=CHECKPOINT_STEP):
    x = [np.array([x_0, y_0])]
    m_values = [np.array([0,0])]
    v_values = [np.array([0,0])]
    timestep = -1
    for t in range(NUM_TIMESTEPS + detection_delay):
        prev_m = m_values[-1]
        prev_v = v_values[-1]
        if strategy != "clean_reset_optim" and strategy != "corr_reset_optim":
            timestep = t
        else:
            timestep += 1
        if t == t_corrupt + (detection_delay + 1):
            if strategy == "clean_clean_corr":
                x[-1] = clean_x[timestep]
                prev_m = clean_m[timestep]
            elif strategy == "clean_corr_clean":
                x[-1] = clean_x[timestep]
                prev_v = clean_v[timestep]
            elif strategy == "clean_corr_corr":
                x[-1] = clean_x[timestep]
            elif strategy == "corr_clean_clean":
                prev_m = clean_m[timestep]
                prev_v = clean_v[timestep]
            elif strategy == "corr_clean_corr":
                prev_m = clean_m[timestep]
            elif strategy == "corr_corr_clean":
                prev_v = clean_v[timestep]
            elif strategy == "rollback":
                if checkpoint_step > t:
                    checkpoint_step = 0
                corrected_x, corrected_m, corrected_v = 0, 0, 0
                for rollback_step in range(0, t - checkpoint_step):
                    if rollback_step == 0:
                        corrected_m = m_values[checkpoint_step]
                        corrected_v = v_values[checkpoint_step]
                        corrected_x = x[checkpoint_step]
                    corrected_gradient = np.array([2*corrected_x[0] + corrected_x[1], corrected_x[0] + 2*corrected_x[1]])
                    corrected_m = beta_1 * corrected_m + (1-beta_1)*corrected_gradient
                    corrected_v = beta_2 * corrected_v + (1-beta_2)*corrected_gradient**2
                    corrected_hat_m = (corrected_m)/(1-beta_1**(checkpoint_step+rollback_step+1))
                    corrected_hat_v = (corrected_v)/(1-beta_2**(checkpoint_step+rollback_step+1))
                    corrected_x = corrected_x - ((alpha)/(np.sqrt(corrected_hat_v) + 10**(-8)))*corrected_hat_m
                x[-1] = corrected_x
                prev_m = corrected_m
                m_values[-1] = corrected_m
                prev_v = corrected_v
                v_values[-1] = corrected_v
            elif strategy == "clean_reset_optim" or strategy == "corr_reset_optim" or strategy == "clean_reset_moments" or strategy == "corr_reset_moments":
                prev_m = np.array([0,0])
                prev_v = np.array([0,0])
            if strategy == "clean_reset_optim" or strategy == "corr_reset_optim":
                timestep = 0
        gradient = np.array([2*x[-1][0] + x[-1][1], x[-1][0] + 2*x[-1][1]])
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset_optim" and strategy != "clean_reset_moments":
            gradient += np.array([delta, 0])
        m = beta_1 * prev_m + (1-beta_1)*gradient
        v = beta_2 * prev_v + (1-beta_2)*gradient**2
        hat_m = (m)/(1-beta_1**(timestep+1))
        hat_v = (v)/(1-beta_2**(timestep+1))
        x.append(x[-1] - ((alpha)/(np.sqrt(hat_v) + 10**(-8)))*hat_m)
        m_values.append(m)
        v_values.append(v)
    return np.array(x), np.array(m_values), np.array(v_values)

def calculate_metrics(clean_x, corrupted_x, t_corrupt, epsilon, detection_delay):
    peak_excess = max(corrupted_x[t_corrupt+1+detection_delay:] - clean_x[t_corrupt+1+detection_delay:])
    cumulative_excess = sum(corrupted_x[t_corrupt+1+detection_delay:] - clean_x[t_corrupt+1+detection_delay:])
    recovery_time = np.where(abs(corrupted_x[t_corrupt+1+detection_delay:] - clean_x[t_corrupt+1+detection_delay:]) > epsilon)[0]
    if len(recovery_time) == 0:
        recovery_time = 0
    elif recovery_time[-1] == NUM_TIMESTEPS - t_corrupt - 1 - detection_delay:
        recovery_time = None
    else:
        recovery_time = recovery_time[-1] + 1
    return peak_excess, cumulative_excess, recovery_time

def normalize_shapley(shap_df, method):
    if method == "raw" or shap_df["optimizer"][0] == "momentum_1d" or shap_df["optimizer"][0] == "momentum_2d":
        return shap_df
    if shap_df['optimizer'][0] == "adam_1d" or shap_df['optimizer'][0] == "adam_2d":
        # adam normalization
        normalize_comp = [4, 5, 6]
        normalize_inter = [7, 8, 9]
    if method == "normalized":
        normalized_shap_df = shap_df.copy()
        comp_attribution = (
            normalized_shap_df.iloc[:, normalize_comp]
            .abs()
            .sum(axis=1)
        )
        inter_attribution = (
            normalized_shap_df.iloc[:, normalize_inter]
            .abs()
            .sum(axis=1)
        )
        normalized_shap_df.iloc[:, normalize_comp] = (
            normalized_shap_df.iloc[:, normalize_comp]
            .div(comp_attribution.replace(0, np.nan), axis=0)
        )
        normalized_shap_df.iloc[:, normalize_inter] = (
            normalized_shap_df.iloc[:, normalize_inter]
            .div(inter_attribution.replace(0, np.nan), axis=0)
        )
    return normalized_shap_df


def save_figure(fig, output_dir, relative_path):
    path = output_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)

def plot_excess_loss(ax, timesteps, loss, clean_loss, detection_delay, **plot_kwargs):
    start = T_CORRUPTION + detection_delay
    ax.plot(
        timesteps[:T_CORRUPTION + 1],
        loss[start:] - clean_loss[start:],
        **plot_kwargs,
    )

def plot_shapley_vs_delay(df, component, optimizer, dimension, normalization):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    fig.suptitle(f"Shapley values for {component}", fontsize=16)
    ax[0].set_title("Negative error")
    ax[1].set_title("Positive error")
    for axis in ax:
        axis.set(
            xlabel="Detection delay",
            ylabel="Shapley value"
        )
        axis.set_xticks(DETECTION_DELAYS)
        if normalization == "raw":
            axis.set_yscale("symlog")
        axis.grid()

    df = normalize_shapley(df, normalization)
    colors = {10:"red", 50:"orange", 100:"green", 1000:"blue"}

    for axis, fig_direction in zip(ax, ["negative", "positive"]):
        filtered_df = df.loc[(df["direction"] == fig_direction)
                            & (df["optimizer"] == f"{optimizer}_{dimension}d")]

        for error, group in filtered_df.groupby("error"):
            group = group.sort_values("detection_delay")

            axis.plot(
                group["detection_delay"],
                group[f"shapley_{component}"],
                label=f"Error = {error}",
                color=colors[error]
            )

        axis.legend()

    fig.tight_layout()
    output_dir = Path(__file__).resolve().parent / "importance_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    if optimizer == "momentum":   
        fig.savefig(output_dir / f"{dimension}d_{optimizer}/{component}.png")
    else:
        fig.savefig(output_dir / f"{dimension}d_{optimizer}/{normalization}/{component}.png")
    plt.close(fig)




def results_frame(results, state_columns):
    rows = []
    for key, metrics in results.items():
        *states, detection_delay, error_sign = key
        rows.append(dict(zip(state_columns, states)) | {
            "detection delay": detection_delay,
            "error sign": error_sign,
            **metrics,
        })
    return pd.DataFrame(rows)


def make_metrics_grabber(results, key_suffix, metric="cumulative excess loss"):
    def grabber(input_matrix):
        values = []
        for row in input_matrix:
            player_state = tuple(bool(value) for value in row)
            values.append(results[player_state + key_suffix][metric])
        return np.asarray(values, dtype=float)

    return grabber
