from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shapiq

from helpers import (
    CHECKPOINT_STEP as checkpoint_step,
    DETECTION_DELAYS as detection_delays,
    NUM_TIMESTEPS as num_timesteps,
    T_CORRUPTION as t_corruption,
    adam_1d,
    adam_2d,
    calculate_metrics,
    make_metrics_grabber,
    momentum_1d,
    momentum_2d,
    normalize_shapley,
    plot_excess_loss,
    plot_shapley_vs_delay,
    results_frame,
    save_figure,
)


"""
Variable initialization
"""
x_0 = 5
y_0 = 5
alpha = 0.1
gamma = 0.9
beta_1 = 0.9
beta_2 = 0.99
errors = [10, 50, 100, 1000]
epsilon = 0.01

t = np.arange(0, num_timesteps+1, 1)

"""
Plotting with different detection delays
"""
shapley_momentum_results = []
shapley_adam_results = []
for error, i in product(errors, range(2)):
    if i == 0:
        delta = error
        label = "positive"
    else:
        delta = -1 * error
        label = "negative"
    output_dir = Path(label) / f"error_{error}"
    momentum_1d_row_accum = {}
    adam_1d_row_accum = {}
    momentum_2d_row_accum = {}
    adam_2d_row_accum = {}

    for detection_delay in detection_delays:
        t = np.arange(0, num_timesteps+detection_delay+1, 1)
        """
        1D experiments without resets
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep since detection", ylabel="Corruption induced excess loss vs. clean run", title=f"1D Momentum (y=x^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_gradients, clean_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "clean", [], [])
        clean_loss = clean_theta**2
        plot_excess_loss(ax, t, clean_loss, clean_loss, detection_delay, label="Clean weights clean velocities", color='black')
        momentum_1d_row_accum[(False, False, False, False, detection_delay, label)] = {"peak excess loss": 0, "cumulative excess loss": 0, "final loss": clean_loss[-1], "recovery time": 0}

        clean_corr_theta, clean_corr_gradients, clean_corr_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "clean_corr", clean_theta, clean_velocities)
        clean_corr_loss = clean_corr_theta**2
        plot_excess_loss(ax, t, clean_corr_loss, clean_loss, detection_delay, label="Clean weights corrupted velocities", color='orange')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum[(False, True, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_corr_loss[-1], "recovery time": recovery_time}

        corr_clean_theta, corr_clean_gradients, corr_clean_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_clean", clean_theta, clean_velocities)
        corr_clean_loss = corr_clean_theta**2
        plot_excess_loss(ax, t, corr_clean_loss, clean_loss, detection_delay, label="Corrupted weights clean velocities", color="green")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum[(True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_clean_loss[-1], "recovery time": recovery_time}

        corr_corr_theta, corr_corr_gradients, corr_corr_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_corr", clean_theta, clean_velocities)
        corr_corr_loss = corr_corr_theta**2
        plot_excess_loss(ax, t, corr_corr_loss, clean_loss, detection_delay, label="Corrupted weights corrupted velocities", color="blue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum[(True, True, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_corr_loss[-1], "recovery time": recovery_time}

        momentum_1d_grabber = make_metrics_grabber(
            momentum_1d_row_accum,
            (False, False, detection_delay, label),
        )
        computer = shapiq.ExactComputer(
            momentum_1d_grabber,
            n_players=2
        )

        sv = computer(index="SV", order=1)
        interactions = computer(index="SII", order=2)
        shapley_momentum_results.append({
            "optimizer": "momentum_1d",
            "error": error,
            "direction": label,
            "detection_delay": detection_delay,
            "shapley_weights": sv[(0,)],
            "shapley_velocity": sv[(1,)],
            "shapley_weights_velocity": interactions[(0, 1)]
        })

        rollback_theta, rollback_gradients, rollback_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "rollback", clean_theta, clean_velocities)
        rollback_loss = rollback_theta**2
        plot_excess_loss(ax, t, rollback_loss, clean_loss, detection_delay, label=f"Rollback (checkpoint {checkpoint_step})", color='purple')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, rollback_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum[(True, True, False, True, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": rollback_loss[-1], "recovery time": recovery_time}

        ax.grid()
        plt.legend()
        save_figure(fig, output_dir, f"1D_momentum/no_reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()


        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep since detection", ylabel="Corruption induced excess loss vs. clean run", title=f"1D Momentum (y=x^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_gradients, clean_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "clean", [], [])
        clean_loss = clean_theta**2
        plot_excess_loss(ax, t, clean_loss, clean_loss, detection_delay, label="Clean weights clean velocities", color='black')
        momentum_1d_row_accum[(False, False, False, False, detection_delay, label)] = {"peak excess loss": 0, "cumulative excess loss": 0, "final loss": clean_loss[-1], "recovery time": 0}

        rollback_theta, _, _ = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "rollback", clean_theta, clean_velocities)
        rollback_loss = rollback_theta**2
        plot_excess_loss(ax, t, rollback_loss, clean_loss, detection_delay, label=f"Rollback (checkpoint {checkpoint_step})", color="purple")

        clean_reset_theta, clean_reset_gradients, clean_reset_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "clean_reset", clean_theta, clean_velocities)
        clean_reset_loss = clean_reset_theta**2
        plot_excess_loss(ax, t, clean_reset_loss, clean_loss, detection_delay, label="Clean weights reset velocities", color='red')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum[(False, False, True, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_reset_loss[-1], "recovery time": recovery_time}

        corr_reset_theta, corr_reset_gradients, corr_reset_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_reset", clean_theta, clean_velocities)
        corr_reset_loss = corr_reset_theta**2
        plot_excess_loss(ax, t, corr_reset_loss, clean_loss, detection_delay, label="Corrupted weights reset velocities", color='lightpink')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum[(True, False, True, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_reset_loss[-1], "recovery time": recovery_time}
        
        ax.grid()
        plt.legend()
        save_figure(fig, output_dir, f"1D_momentum/reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep since detection", ylabel="Corruption induced excess loss vs. clean run", title=f"1D Adam (y=x^2) after {label} corruption (detection delay = {detection_delay})")

        # 1d example (y = x^2) with Adam - no corruption
        clean_theta, clean_m, clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean", [], [], [])
        clean_loss = clean_theta**2
        plot_excess_loss(ax, t, clean_loss, clean_loss, detection_delay, label="Clean weights clean m clean v", color='black')
        adam_1d_row_accum[(False, False, False, False, False, False, detection_delay, label)] = {"peak excess loss": 0, "cumulative excess loss": 0, "final loss": clean_loss[-1], "recovery time": 0}

        clean_clean_corr_theta, clean_clean_corr_m, clean_clean_corr_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_clean_corr", clean_theta, clean_m, clean_v)
        clean_clean_corr_loss = clean_clean_corr_theta**2
        plot_excess_loss(ax, t, clean_clean_corr_loss, clean_loss, detection_delay, label="Clean weights clean m corr v", color="orange")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_clean_corr_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(False, False, True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_clean_corr_loss[-1], "recovery time": recovery_time}

        clean_corr_clean_theta, clean_corr_clean_m, clean_corr_clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_corr_clean", clean_theta, clean_m, clean_v)
        clean_corr_clean_loss = clean_corr_clean_theta**2
        plot_excess_loss(ax, t, clean_corr_clean_loss, clean_loss, detection_delay, label="Clean weights corr m clean v", color="olive")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_clean_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(False, True, False, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_corr_clean_loss[-1], "recovery time": recovery_time}

        clean_corr_corr_theta, clean_corr_corr_m, clean_corr_corr_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_corr_corr", clean_theta, clean_m, clean_v)
        clean_corr_corr_loss = clean_corr_corr_theta**2
        plot_excess_loss(ax, t, clean_corr_corr_loss, clean_loss, detection_delay, label="Clean weights corr m corr v", color="green")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_corr_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(False, True, True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_corr_corr_loss[-1], "recovery time": recovery_time}

        corr_clean_clean_theta, corr_clean_clean_m, corr_clean_clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_clean_clean", clean_theta, clean_m, clean_v)
        corr_clean_clean_loss = corr_clean_clean_theta**2
        plot_excess_loss(ax, t, corr_clean_clean_loss, clean_loss, detection_delay, label="Corr weights clean m clean v", color="cyan")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_clean_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(True, False, False, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_clean_clean_loss[-1], "recovery time": recovery_time}

        corr_clean_corr_theta, corr_clean_corr_m, corr_clean_corr_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_clean_corr", clean_theta, clean_m, clean_v)
        corr_clean_corr_loss = corr_clean_corr_theta**2
        plot_excess_loss(ax, t, corr_clean_corr_loss, clean_loss, detection_delay, label="Corr weights clean m corr v", color="blue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_corr_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(True, False, True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_clean_corr_loss[-1], "recovery time": recovery_time}

        corr_corr_clean_theta, corr_corr_clean_m, corr_corr_clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_corr_clean", clean_theta, clean_m, clean_v)
        corr_corr_clean_loss = corr_corr_clean_theta**2
        plot_excess_loss(ax, t, corr_corr_clean_loss, clean_loss, detection_delay, label="Corr weights corr m clean v", color="purple")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_clean_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(True, True, False, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_corr_clean_loss[-1], "recovery time": recovery_time}

        corr_corr_corr_theta, corr_corr_corr_m, corr_corr_corr_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_corr_corr", clean_theta, clean_m, clean_v)
        corr_corr_corr_loss = corr_corr_corr_theta**2
        plot_excess_loss(ax, t, corr_corr_corr_loss, clean_loss, detection_delay, label="Corr weights corr m corr v", color="pink")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_corr_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(True, True, True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_corr_corr_loss[-1], "recovery time": recovery_time}

        adam_1d_grabber = make_metrics_grabber(
            adam_1d_row_accum,
            (False, False, False, detection_delay, label),
        )
        computer = shapiq.ExactComputer(
            adam_1d_grabber,
            n_players=3
        )

        sv = computer(index="SV", order=1)
        interactions = computer(index="SII", order=2)
        shapley_adam_results.append({
            "optimizer": "adam_1d",
            "error": error,
            "direction": label,
            "detection_delay": detection_delay,
            "shapley_weights": sv[(0,)],
            "shapley_m": sv[(1,)],
            "shapley_v": sv[(2,)],
            "shapley_weights_m": interactions[(0, 1)],
            "shapley_m_v": interactions[(1, 2)],
            "shapley_weights_v": interactions[(0, 2)],
        })

        rollback_theta, rollback_m, rollback_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "rollback", clean_theta, clean_m, clean_v)
        rollback_loss = rollback_theta**2
        plot_excess_loss(ax, t, rollback_loss, clean_loss, detection_delay, label="Rollback", color="midnightblue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, rollback_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(True, True, True, False, True, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": rollback_loss[-1], "recovery time": recovery_time}

        ax.grid()
        plt.legend()
        save_figure(fig, output_dir, f"1D_adam/no_reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep since detection", ylabel="Corruption induced excess loss vs. clean run", title=f"1D Adam (y=x^2) after {label} corruption (detection delay = {detection_delay})")

        # 1d example (y = x^2) with Adam - no corruption
        clean_theta, clean_m, clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean", [], [], [])
        clean_loss = clean_theta**2
        plot_excess_loss(ax, t, clean_loss, clean_loss, detection_delay, label="Clean weights clean m clean v", color='black')
        adam_1d_row_accum[(False, False, False, False, False, False, detection_delay, label)] = {"peak excess loss": 0, "cumulative excess loss": 0, "final loss": clean_loss[-1], "recovery time": 0}

        rollback_theta, _, _ = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "rollback", clean_theta, clean_m, clean_v)
        rollback_loss = rollback_theta**2
        plot_excess_loss(ax, t, rollback_loss, clean_loss, detection_delay, label=f"Rollback (checkpoint {checkpoint_step})", color="midnightblue")

        clean_reset_moments_theta, clean_reset_moments_m, clean_reset_moments_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_reset_moments", clean_theta, clean_m, clean_v)
        clean_reset_moments_loss = clean_reset_moments_theta**2
        plot_excess_loss(ax, t, clean_reset_moments_loss, clean_loss, detection_delay, label="Clean weights reset m reset v", color='red')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_moments_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(False, False, False, True, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_reset_moments_loss[-1], "recovery time": recovery_time}

        clean_reset_optim_theta, clean_reset_optim_m, clean_reset_optim_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_reset_optim", clean_theta, clean_m, clean_v)
        clean_reset_optim_loss = clean_reset_optim_theta**2
        plot_excess_loss(ax, t, clean_reset_optim_loss, clean_loss, detection_delay, label="Clean weights reset m reset v restart optimizer", color='maroon')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_optim_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(False, False, False, True, False, True, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_reset_optim_loss[-1], "recovery time": recovery_time}

        corr_reset_moments_theta, corr_reset_moments_m, corr_reset_moments_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_reset_moments", clean_theta, clean_m, clean_v)
        corr_reset_moments_loss = corr_reset_moments_theta**2
        plot_excess_loss(ax, t, corr_reset_moments_loss, clean_loss, detection_delay, label="Corrupted weights reset m reset v", color='darkorange')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_moments_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(True, False, False, True, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_reset_moments_loss[-1], "recovery time": recovery_time}

        corr_reset_optim_theta, corr_reset_optim_m, corr_reset_optim_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_reset_optim", clean_theta, clean_m, clean_v)
        corr_reset_optim_loss = corr_reset_optim_theta**2
        plot_excess_loss(ax, t, corr_reset_optim_loss, clean_loss, detection_delay, label="Corrupted weights reset m reset v restart optimizer", color='lightpink')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_optim_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum[(True, False, False, True, False, True, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_reset_optim_loss[-1], "recovery time": recovery_time}

        ax.grid()
        plt.legend()
        save_figure(fig, output_dir, f"1D_adam/reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()

        """
        2D experiments
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep since detection", ylabel="Corruption induced excess loss vs. clean run", title=f"2D Momentum (f(x,y) = x^2 + xy + y^2) after {label} corruption (detection delay = {detection_delay})")

        # 2d example (f(x, y) = x^2 + xy + y^2) with momentum - no corruption
        clean_theta, clean_gradients, clean_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "clean", [], [])
        clean_loss = clean_theta[:,0]**2 + clean_theta[:,0]*clean_theta[:,1] + clean_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_loss, clean_loss, detection_delay, label="Clean weights clean velocities", color='black')
        momentum_2d_row_accum[(False, False, False, False, detection_delay, label)] = {"peak excess loss": 0, "cumulative excess loss": 0, "final loss": clean_loss[-1], "recovery time": 0}

        clean_corr_theta, clean_corr_gradients, clean_corr_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "clean_corr", clean_theta, clean_velocities)
        clean_corr_loss = clean_corr_theta[:,0]**2 + clean_corr_theta[:,0]*clean_corr_theta[:,1] + clean_corr_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_corr_loss, clean_loss, detection_delay, label="Clean weights corr velocities", color='orange')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum[(False, True, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_corr_loss[-1], "recovery time": recovery_time}

        corr_clean_theta, corr_clean_gradients, corr_clean_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_clean", clean_theta, clean_velocities)
        corr_clean_loss = corr_clean_theta[:,0]**2 + corr_clean_theta[:,0]*corr_clean_theta[:,1] + corr_clean_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_clean_loss, clean_loss, detection_delay, label="Corr weights clean velocities", color="green")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum[(True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_clean_loss[-1], "recovery time": recovery_time}
        
        corr_corr_theta, corr_corr_gradients, corr_corr_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_corr", clean_theta, clean_velocities)
        corr_corr_loss = corr_corr_theta[:,0]**2 + corr_corr_theta[:,0]*corr_corr_theta[:,1] + corr_corr_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_corr_loss, clean_loss, detection_delay, label="Corr weights corr velocities", color="blue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum[(True, True, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_corr_loss[-1], "recovery time": recovery_time}

        momentum_2d_grabber = make_metrics_grabber(
            momentum_2d_row_accum,
            (False, False, detection_delay, label),
        )
        computer = shapiq.ExactComputer(
            momentum_2d_grabber,
            n_players=2
        )

        sv = computer(index="SV", order=1)
        interactions = computer(index="SII", order=2)
        shapley_momentum_results.append({
            "optimizer": "momentum_2d",
            "error": error,
            "direction": label,
            "detection_delay": detection_delay,
            "shapley_weights": sv[(0,)],
            "shapley_velocity": sv[(1,)],
            "shapley_weights_velocity": interactions[(0, 1)]
        })

        rollback_theta, rollback_gradients, rollback_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "rollback", clean_theta, clean_velocities)
        rollback_loss = rollback_theta[:,0]**2 + rollback_theta[:,0]*rollback_theta[:,1] + rollback_theta[:,1]**2
        plot_excess_loss(ax, t, rollback_loss, clean_loss, detection_delay, label=f"Rollback (checkpoint {checkpoint_step})", color="purple")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, rollback_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum[(True, True, False, True, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": rollback_loss[-1], "recovery time": recovery_time}

        ax.grid()
        plt.legend()
        save_figure(fig, output_dir, f"2D_momentum/no_reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()


        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep since detection", ylabel="Corruption induced excess loss vs. clean run", title=f"2D Momentum (f(x,y) = x^2 + xy + y^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_gradients, clean_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "clean", [], [])
        clean_loss = clean_theta[:,0]**2 + clean_theta[:,0]*clean_theta[:,1] + clean_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_loss, clean_loss, detection_delay, label="Clean weights clean velocities", color='black')
        momentum_2d_row_accum[(False, False, False, False, detection_delay, label)] = {"peak excess loss": 0, "cumulative excess loss": 0, "final loss": clean_loss[-1], "recovery time": 0}

        rollback_theta, _, _ = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "rollback", clean_theta, clean_velocities)
        rollback_loss = rollback_theta[:,0]**2 + rollback_theta[:,0]*rollback_theta[:,1] + rollback_theta[:,1]**2
        plot_excess_loss(ax, t, rollback_loss, clean_loss, detection_delay, label=f"Rollback (checkpoint {checkpoint_step})", color="purple")

        clean_reset_theta, clean_reset_gradients, clean_reset_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "clean_reset", clean_theta, clean_velocities)
        clean_reset_loss = clean_reset_theta[:,0]**2 + clean_reset_theta[:,0]*clean_reset_theta[:,1] + clean_reset_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_reset_loss, clean_loss, detection_delay, label="Clean weights reset velocities", color='red')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum[(False, False, True, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_reset_loss[-1], "recovery time": recovery_time}

        corr_reset_theta, corr_reset_gradients, corr_reset_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_reset", clean_theta, clean_velocities)  
        corr_reset_loss = corr_reset_theta[:,0]**2 + corr_reset_theta[:,0]*corr_reset_theta[:,1] + corr_reset_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_reset_loss, clean_loss, detection_delay, label="Corrupted weights reset velocities", color='lightpink')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum[(True, False, True, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_reset_loss[-1], "recovery time": recovery_time}

        ax.grid()
        plt.legend()
        save_figure(fig, output_dir, f"2D_momentum/reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()


        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep since detection", ylabel="Corruption induced excess loss vs. clean run", title=f"2D Adam (f(x,y) = x^2+xy+y^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_m, clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean", [], [], [])
        clean_loss = clean_theta[:,0]**2 + clean_theta[:,0]*clean_theta[:,1] + clean_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_loss, clean_loss, detection_delay, label="Clean weights clean m clean v", color='black')
        adam_2d_row_accum[(False, False, False, False, False, False, detection_delay, label)] = {"peak excess loss": 0, "cumulative excess loss": 0, "final loss": clean_loss[-1], "recovery time": 0}

        clean_clean_corr_theta, clean_clean_corr_m, clean_clean_corr_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_clean_corr", clean_theta, clean_m, clean_v)
        clean_clean_corr_loss = clean_clean_corr_theta[:,0]**2 + clean_clean_corr_theta[:,0]*clean_clean_corr_theta[:,1] + clean_clean_corr_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_clean_corr_loss, clean_loss, detection_delay, label="Clean weights clean m corr v", color="orange")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_clean_corr_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(False, False, True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_clean_corr_loss[-1], "recovery time": recovery_time}

        clean_corr_clean_theta, clean_corr_clean_m, clean_corr_clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_corr_clean", clean_theta, clean_m, clean_v)
        clean_corr_clean_loss = clean_corr_clean_theta[:,0]**2 + clean_corr_clean_theta[:,0]*clean_corr_clean_theta[:,1] + clean_corr_clean_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_corr_clean_loss, clean_loss, detection_delay, label="Clean weights corr m clean v", color="olive")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_clean_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(False, True, False, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_corr_clean_loss[-1], "recovery time": recovery_time}

        clean_corr_corr_theta, clean_corr_corr_m, clean_corr_corr_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_corr_corr", clean_theta, clean_m, clean_v)
        clean_corr_corr_loss = clean_corr_corr_theta[:,0]**2 + clean_corr_corr_theta[:,0]*clean_corr_corr_theta[:,1] + clean_corr_corr_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_corr_corr_loss, clean_loss, detection_delay, label="Clean weights corr m corr v", color="green")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_corr_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(False, True, True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_corr_corr_loss[-1], "recovery time": recovery_time}

        corr_clean_clean_theta, corr_clean_clean_m, corr_clean_clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_clean_clean", clean_theta, clean_m, clean_v)
        corr_clean_clean_loss = corr_clean_clean_theta[:,0]**2 + corr_clean_clean_theta[:,0]*corr_clean_clean_theta[:,1] + corr_clean_clean_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_clean_clean_loss, clean_loss, detection_delay, label="Corr weights clean m clean v", color="cyan")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_clean_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(True, False, False, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_clean_clean_loss[-1], "recovery time": recovery_time}

        corr_clean_corr_theta, corr_clean_corr_m, corr_clean_corr_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_clean_corr", clean_theta, clean_m, clean_v)
        corr_clean_corr_loss = corr_clean_corr_theta[:,0]**2 + corr_clean_corr_theta[:,0]*corr_clean_corr_theta[:,1] + corr_clean_corr_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_clean_corr_loss, clean_loss, detection_delay, label="Corr weights clean m corr v", color="blue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_corr_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(True, False, True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_clean_corr_loss[-1], "recovery time": recovery_time}

        corr_corr_clean_theta, corr_corr_clean_m, corr_corr_clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_corr_clean", clean_theta, clean_m, clean_v)
        corr_corr_clean_loss = corr_corr_clean_theta[:,0]**2 + corr_corr_clean_theta[:,0]*corr_corr_clean_theta[:,1] + corr_corr_clean_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_corr_clean_loss, clean_loss, detection_delay, label="Corr weights corr m clean v", color="purple")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_clean_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(True, True, False, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_corr_clean_loss[-1], "recovery time": recovery_time}

        corr_corr_corr_theta, corr_corr_corr_m, corr_corr_corr_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_corr_corr", clean_theta, clean_m, clean_v)
        corr_corr_corr_loss = corr_corr_corr_theta[:,0]**2 + corr_corr_corr_theta[:,0]*corr_corr_corr_theta[:,1] + corr_corr_corr_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_corr_corr_loss, clean_loss, detection_delay, label="Corr weights corr m corr v", color="pink")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_corr_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(True, True, True, False, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_corr_corr_loss[-1], "recovery time": recovery_time}

        adam_2d_grabber = make_metrics_grabber(
            adam_2d_row_accum,
            (False, False, False, detection_delay, label),
        )
        computer = shapiq.ExactComputer(
            adam_2d_grabber,
            n_players=3
        )

        sv = computer(index="SV", order=1)
        interactions = computer(index="SII", order=2)
        shapley_adam_results.append({
            "optimizer": "adam_2d",
            "error": error,
            "direction": label,
            "detection_delay": detection_delay,
            "shapley_weights": sv[(0,)],
            "shapley_m": sv[(1,)],
            "shapley_v": sv[(2,)],
            "shapley_weights_m": interactions[(0, 1)],
            "shapley_m_v": interactions[(1, 2)],
            "shapley_weights_v": interactions[(0, 2)],
        })

        rollback_theta, rollback_m, rollback_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "rollback", clean_theta, clean_m, clean_v)
        rollback_loss = rollback_theta[:,0]**2 + rollback_theta[:,0]*rollback_theta[:,1] + rollback_theta[:,1]**2
        plot_excess_loss(ax, t, rollback_loss, clean_loss, detection_delay, label=f"Rollback (checkpoint {checkpoint_step})", color="midnightblue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, rollback_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(True, True, True, False, True, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": rollback_loss[-1], "recovery time": recovery_time}

        ax.grid()
        plt.legend()
        save_figure(fig, output_dir, f"2D_adam/no_reset/{detection_delay}_step_delay.png")
        plt.close()
        # plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep since detection", ylabel="Corruption induced excess loss vs. clean run", title=f"2D Adam (f(x,y) = x^2+xy+y^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_m, clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean", [], [], [])
        clean_loss = clean_theta[:,0]**2 + clean_theta[:,0]*clean_theta[:,1] + clean_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_loss, clean_loss, detection_delay, label="Clean weights clean m clean v", color='black')
        adam_2d_row_accum[(False, False, False, False, False, False, detection_delay, label)] = {"peak excess loss": 0, "cumulative excess loss": 0, "final loss": clean_loss[-1], "recovery time": 0}

        rollback_theta, _, _ = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "rollback", clean_theta, clean_m, clean_v)
        rollback_loss = rollback_theta[:,0]**2 + rollback_theta[:,0]*rollback_theta[:,1] + rollback_theta[:,1]**2
        plot_excess_loss(ax, t, rollback_loss, clean_loss, detection_delay, label=f"Rollback (checkpoint {checkpoint_step})", color="midnightblue")

        clean_reset_moments_theta, clean_reset_moments_m, clean_reset_moments_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_reset_moments", clean_theta, clean_m, clean_v)
        clean_reset_moments_loss = clean_reset_moments_theta[:,0]**2 + clean_reset_moments_theta[:,0]*clean_reset_moments_theta[:,1] + clean_reset_moments_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_reset_moments_loss, clean_loss, detection_delay, label="Clean weights reset m reset v", color='red')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_moments_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(False, False, False, True, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_reset_moments_loss[-1], "recovery time": recovery_time}

        clean_reset_optim_theta, clean_reset_optim_m, clean_reset_optim_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_reset_optim", clean_theta, clean_m, clean_v)
        clean_reset_optim_loss = clean_reset_optim_theta[:,0]**2 + clean_reset_optim_theta[:,0]*clean_reset_optim_theta[:,1] + clean_reset_optim_theta[:, 1]**2
        plot_excess_loss(ax, t, clean_reset_optim_loss, clean_loss, detection_delay, label="Clean weights reset m reset v restart optim", color='maroon')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_optim_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(False, False, False, True, False, True, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": clean_reset_optim_loss[-1], "recovery time": recovery_time}

        corr_reset_moments_theta, corr_reset_m, corr_reset_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_reset_moments", clean_theta, clean_m, clean_v)
        corr_reset_loss = corr_reset_moments_theta[:,0]**2 + corr_reset_moments_theta[:,0]*corr_reset_moments_theta[:,1] + corr_reset_moments_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_reset_loss, clean_loss, detection_delay, label="Corrupted weights reset m reset v", color='darkorange')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(True, False, False, True, False, False, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_reset_loss[-1], "recovery time": recovery_time}

        corr_reset_optim_theta, corr_reset_optim_m, corr_reset_optim_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_reset_optim", clean_theta, clean_m, clean_v)
        corr_reset_optim_loss = corr_reset_optim_theta[:,0]**2 + corr_reset_optim_theta[:,0]*corr_reset_optim_theta[:,1] + corr_reset_optim_theta[:, 1]**2
        plot_excess_loss(ax, t, corr_reset_optim_loss, clean_loss, detection_delay, label="Corrupted weights reset m reset v restart optim", color='lightpink')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_optim_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum[(True, False, False, True, False, True, detection_delay, label)] = {"peak excess loss": peak_excess, "cumulative excess loss": cumulative_excess, "final loss": corr_reset_optim_loss[-1], "recovery time": recovery_time}

        ax.grid()
        plt.legend()
        save_figure(fig, output_dir, f"2D_adam/reset/{detection_delay}_step_delay.png")
        plt.close()
        # plt.show()


    momentum_columns = ["corrupted weights?", "corrupted velocities?", "reset?", "rollback?"]
    adam_columns = ["corrupted weights?", "corrupted m?", "corrupted v?", "reset?", "rollback?", "restarted optim?"]
    momentum_1d_results = results_frame(momentum_1d_row_accum, momentum_columns)
    adam_1d_results = results_frame(adam_1d_row_accum, adam_columns)
    momentum_2d_results = results_frame(momentum_2d_row_accum, momentum_columns)
    adam_2d_results = results_frame(adam_2d_row_accum, adam_columns)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    momentum_1d_results.to_csv(results_dir / "1D_momentum_results.csv", index=False)
    adam_1d_results.to_csv(results_dir / "1D_adam_results.csv", index=False)
    momentum_2d_results.to_csv(results_dir / "2D_momentum_results.csv", index=False)
    adam_2d_results.to_csv(results_dir / "2D_adam_results.csv", index=False)

pd.DataFrame(shapley_momentum_results).to_csv(
    "importance_analysis/shapley_momentum_results.csv",
    index=False
)

pd.DataFrame(shapley_adam_results).to_csv(
    "importance_analysis/shapley_adam_results.csv",
    index=False
)

for optimizer, components in [("momentum", ["weights", "velocity", "weights_velocity"]), ("adam",["weights", "m", "v", "weights_m", "m_v", "weights_v"])]:
    if optimizer == "adam":
        shapley_df = pd.DataFrame(shapley_adam_results)
    else:
        shapley_df = pd.DataFrame(shapley_momentum_results)
    normalization = "raw"
    for dimension in [1, 2]:
        for component in components:
            plot_shapley_vs_delay(shapley_df, component, optimizer, dimension, normalization)
    if optimizer == "adam":
        normalization = "normalized"
        for dimension in [1, 2]:
            for component in components:
                plot_shapley_vs_delay(shapley_df, component, optimizer, dimension, normalization)
