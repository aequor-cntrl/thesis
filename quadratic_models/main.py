import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Variable initialization
"""
x_0 = 5
y_0 = 5
alpha = 0.1
gamma = 0.9
beta_1 = 0.9
beta_2 = 0.99
error = 100
t_corruption = 50
detection_delays = [0, 4, 16]
num_timesteps = 100
epsilon = 0.01

t = np.arange(0, num_timesteps+1, 1)

"""
Functions
"""
def momentum_1d(x_0, alpha, gamma, t_corrupt, detection_delay, delta, strategy, clean_x, clean_velocities):
    x = [x_0]
    gradients = [0]
    velocities = [0]
    for t in range(num_timesteps):
        prev_velocity = velocities[-1]
        if t == t_corrupt + (detection_delay + 1):
            if strategy == "corr_clean":
                prev_velocity = clean_velocities[t]
            elif strategy == "clean_corr":
                x[-1] = clean_x[t]
            elif strategy == "clean_reset" or strategy == "corr_reset":
                prev_velocity = 0
        gradient = 2 * x[-1]
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset": 
            gradient += delta
        velocity = gamma * prev_velocity + alpha * gradient
        x.append(x[-1] - velocity)
        gradients.append(gradient)
        velocities.append(velocity)
    return np.array(x), gradients, velocities

def momentum_2d(x_0, y_0, alpha, gamma, t_corrupt, detection_delay, delta, strategy, clean_x, clean_velocities):
    x = [np.array([x_0, y_0])]
    gradients = [np.array([0,0])]
    velocities = [np.array([0,0])]
    for t in range(num_timesteps):
        prev_velocity = velocities[-1]
        if t == t_corrupt + (detection_delay + 1):
            if strategy == "corr_clean":
                prev_velocity = clean_velocities[t]
            elif strategy == "clean_corr":
                x[-1] = clean_x[t]
            elif strategy == "clean_reset" or strategy == "corr_reset":
                prev_velocity = np.array([0,0])
        gradient = np.array([2*x[-1][0] + x[-1][1], x[-1][0] + 2*x[-1][1]])
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset":
            gradient += np.array([delta, 0])
        velocity = gamma * prev_velocity + alpha * gradient
        x.append(x[-1] - velocity)
        gradients.append(gradient)
        velocities.append(velocity)
    return np.array(x), np.array(gradients), np.array(velocities)

def adam_1d(x_0, alpha, beta_1, beta_2, t_corrupt, detection_delay, delta, strategy, clean_x, clean_m, clean_v):
    x = [x_0]
    m_values = [0]
    v_values = [0]
    timestep = -1
    for t in range(num_timesteps):
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

def adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corrupt, detection_delay, delta, strategy, clean_x, clean_m, clean_v):
    x = [np.array([x_0, y_0])]
    m_values = [np.array([0,0])]
    v_values = [np.array([0,0])]
    timestep = -1
    for t in range(num_timesteps):
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
    elif recovery_time[-1] == num_timesteps - t_corrupt - 1 - detection_delay:
        recovery_time = None
    else:
        recovery_time = recovery_time[-1] + 1
    return peak_excess, cumulative_excess, recovery_time

"""
Plotting with different detection delays
"""
for i in range(2):
    if i == 0:
        delta = error
        label = "positive"
    else:
        delta = -1 * error
        label = "negative"
    momentum_1d_row_accum = []
    adam_1d_row_accum = []
    momentum_2d_row_accum = []
    adam_2d_row_accum = []

    for detection_delay in detection_delays:

        """
        1D experiments without resets
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep", ylabel="Loss values", title=f"1D Momentum (y=x^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_gradients, clean_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "clean", [], [])
        clean_loss = clean_theta**2
        ax.plot(t[t_corruption:], clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean velocities", color='black')
        momentum_1d_row_accum.append([False, False, False, detection_delay, 0, 0, clean_loss[-1], 0])

        clean_corr_theta, clean_corr_gradients, clean_corr_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "clean_corr", clean_theta, clean_velocities)
        clean_corr_loss = clean_corr_theta**2
        ax.plot(t[t_corruption:], clean_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights corrupted velocities", color='orange')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum.append([False, True, False, detection_delay, peak_excess, cumulative_excess, clean_corr_loss[-1], recovery_time])

        corr_clean_theta, corr_clean_gradients, corr_clean_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_clean", clean_theta, clean_velocities)
        corr_clean_loss = corr_clean_theta**2
        ax.plot(t[t_corruption:], corr_clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Corrupted weights clean velocities", color="green")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum.append([True, False, False, detection_delay, peak_excess, cumulative_excess, corr_clean_loss[-1], recovery_time])

        corr_corr_theta, corr_corr_gradients, corr_corr_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_corr", clean_theta, clean_velocities)
        corr_corr_loss = corr_corr_theta**2
        ax.plot(t[t_corruption:], corr_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Corrupted weights corrupted velocities", color="blue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum.append([True, True, False, detection_delay, peak_excess, cumulative_excess, corr_corr_loss[-1], recovery_time])

        ax.grid()
        plt.legend()
        fig.savefig(f"{label}/1D_momentum/no_reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep", ylabel="Loss values", title=f"1D Momentum (y=x^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_gradients, clean_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "clean", [], [])
        clean_loss = clean_theta**2
        ax.plot(t[t_corruption:], clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean velocities", color='black')
        momentum_1d_row_accum.append([False, False, False, detection_delay, 0, 0, clean_loss[-1], 0])

        clean_reset_theta, clean_reset_gradients, clean_reset_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "clean_reset", clean_theta, clean_velocities)
        clean_reset_loss = clean_reset_theta**2
        ax.plot(t[t_corruption:], clean_reset_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights reset velocities", color='red')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum.append([False, False, True, detection_delay, peak_excess, cumulative_excess, clean_reset_loss[-1], recovery_time])

        corr_reset_theta, corr_reset_gradients, corr_reset_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_reset", clean_theta, clean_velocities)
        corr_reset_loss = corr_reset_theta**2
        ax.plot(t[t_corruption:], corr_reset_loss[t_corruption:]-clean_loss[t_corruption:], label="Corrupted weights reset velocities", color='lightpink')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_loss, t_corruption, epsilon, detection_delay)
        momentum_1d_row_accum.append([True, False, True, detection_delay, peak_excess, cumulative_excess, corr_reset_loss[-1], recovery_time])
        
        ax.grid()
        plt.legend()
        fig.savefig(f"{label}/1D_momentum/reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep", ylabel="Loss values", title=f"1D Adam (y=x^2) after {label} corruption (detection delay = {detection_delay})")

        # 1d example (y = x^2) with Adam - no corruption
        clean_theta, clean_m, clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean", [], [], [])
        clean_loss = clean_theta**2
        ax.plot(t[t_corruption:], clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean m clean v", color='black')
        adam_1d_row_accum.append([False, False, False, False, False, detection_delay, 0, 0, clean_loss[-1], 0])

        clean_clean_corr_theta, clean_clean_corr_m, clean_clean_corr_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_clean_corr", clean_theta, clean_m, clean_v)
        clean_clean_corr_loss = clean_clean_corr_theta**2
        ax.plot(t[t_corruption:], clean_clean_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean m corr v", color="orange")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_clean_corr_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([False, False, True, False, False, detection_delay, peak_excess, cumulative_excess, clean_clean_corr_loss[-1], recovery_time])

        clean_corr_clean_theta, clean_corr_clean_m, clean_corr_clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_corr_clean", clean_theta, clean_m, clean_v)
        clean_corr_clean_loss = clean_corr_clean_theta**2
        ax.plot(t[t_corruption:], clean_corr_clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights corr m clean v", color="olive")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_clean_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([False, True, False, False, False, detection_delay, peak_excess, cumulative_excess, clean_corr_clean_loss[-1], recovery_time])

        clean_corr_corr_theta, clean_corr_corr_m, clean_corr_corr_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_corr_corr", clean_theta, clean_m, clean_v)
        clean_corr_corr_loss = clean_corr_corr_theta**2
        ax.plot(t[t_corruption:], clean_corr_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights corr m corr v", color="green")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_corr_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([False, True, True, False, False, detection_delay, peak_excess, cumulative_excess, clean_corr_corr_loss[-1], recovery_time])

        corr_clean_clean_theta, corr_clean_clean_m, corr_clean_clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_clean_clean", clean_theta, clean_m, clean_v)
        corr_clean_clean_loss = corr_clean_clean_theta**2
        ax.plot(t[t_corruption:], corr_clean_clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights clean m clean v", color="cyan")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_clean_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([True, False, False, False, False, detection_delay, peak_excess, cumulative_excess, corr_clean_clean_loss[-1], recovery_time])

        corr_clean_corr_theta, corr_clean_corr_m, corr_clean_corr_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_clean_corr", clean_theta, clean_m, clean_v)
        corr_clean_corr_loss = corr_clean_corr_theta**2
        ax.plot(t[t_corruption:], corr_clean_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights clean m corr v", color="blue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_corr_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([True, False, True, False, False, detection_delay, peak_excess, cumulative_excess, corr_clean_corr_loss[-1], recovery_time])

        corr_corr_clean_theta, corr_corr_clean_m, corr_corr_clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_corr_clean", clean_theta, clean_m, clean_v)
        corr_corr_clean_loss = corr_corr_clean_theta**2
        ax.plot(t[t_corruption:], corr_corr_clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights corr m clean v", color="purple")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_clean_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([True, True, False, False, False, detection_delay, peak_excess, cumulative_excess, corr_corr_clean_loss[-1], recovery_time])

        corr_corr_corr_theta, corr_corr_corr_m, corr_corr_corr_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_corr_corr", clean_theta, clean_m, clean_v)
        corr_corr_corr_loss = corr_corr_corr_theta**2
        ax.plot(t[t_corruption:], corr_corr_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights corr m corr v", color="pink")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_corr_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([True, True, True, False, False, detection_delay, peak_excess, cumulative_excess, corr_corr_corr_loss[-1], recovery_time])

        ax.grid()
        plt.legend()
        fig.savefig(f"{label}/1D_adam/no_reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep", ylabel="Loss values", title=f"1D Adam (y=x^2) after {label} corruption (detection delay = {detection_delay})")

        # 1d example (y = x^2) with Adam - no corruption
        clean_theta, clean_m, clean_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean", [], [], [])
        clean_loss = clean_theta**2
        ax.plot(t[t_corruption:], clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean m clean v", color='black')
        adam_1d_row_accum.append([False, False, False, False, False, detection_delay, 0, 0, clean_loss[-1], 0])

        clean_reset_moments_theta, clean_reset_moments_m, clean_reset_moments_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_reset_moments", clean_theta, clean_m, clean_v)
        clean_reset_moments_loss = clean_reset_moments_theta**2
        ax.plot(t[t_corruption:], clean_reset_moments_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights reset m reset v", color='red')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_moments_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([False, False, False, True, False, detection_delay, peak_excess, cumulative_excess, clean_reset_moments_loss[-1], recovery_time])

        clean_reset_optim_theta, clean_reset_optim_m, clean_reset_optim_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_reset_optim", clean_theta, clean_m, clean_v)
        clean_reset_optim_loss = clean_reset_optim_theta**2
        ax.plot(t[t_corruption:], clean_reset_optim_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights reset m reset v restart optimizer", color='maroon')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_optim_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([False, False, False, True, True, detection_delay, peak_excess, cumulative_excess, clean_reset_optim_loss[-1], recovery_time])

        corr_reset_moments_theta, corr_reset_moments_m, corr_reset_moments_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_reset_moments", clean_theta, clean_m, clean_v)
        corr_reset_moments_loss = corr_reset_moments_theta**2
        ax.plot(t[t_corruption:], corr_reset_moments_loss[t_corruption:]-clean_loss[t_corruption:], label="Corrupted weights reset m reset v", color='darkorange')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_moments_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([True, False, False, True, False, detection_delay, peak_excess, cumulative_excess, corr_reset_moments_loss[-1], recovery_time])

        corr_reset_optim_theta, corr_reset_optim_m, corr_reset_optim_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_reset_optim", clean_theta, clean_m, clean_v)
        corr_reset_optim_loss = corr_reset_optim_theta**2
        ax.plot(t[t_corruption:], corr_reset_optim_loss[t_corruption:]-clean_loss[t_corruption:], label="Corrupted weights reset m reset v restart optimizer", color='lightpink')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_optim_loss, t_corruption, epsilon, detection_delay)
        adam_1d_row_accum.append([True, False, False, True, True, detection_delay, peak_excess, cumulative_excess, corr_reset_optim_loss[-1], recovery_time])

        ax.grid()
        plt.legend()
        fig.savefig(f"{label}/1D_adam/reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()

        """
        2D experiments
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep", ylabel="Loss values", title=f"2D Momentum (f(x,y) = x^2 + xy + y^2) after {label} corruption (detection delay = {detection_delay})")

        # 2d example (f(x, y) = x^2 + xy + y^2) with momentum - no corruption
        clean_theta, clean_gradients, clean_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "clean", [], [])
        clean_loss = clean_theta[:,0]**2 + clean_theta[:,0]*clean_theta[:,1] + clean_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean velocities", color='black')
        momentum_2d_row_accum.append([False, False, False, detection_delay, 0, 0, clean_loss[-1], 0])

        clean_corr_theta, clean_corr_gradients, clean_corr_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "clean_corr", clean_theta, clean_velocities)
        clean_corr_loss = clean_corr_theta[:,0]**2 + clean_corr_theta[:,0]*clean_corr_theta[:,1] + clean_corr_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights corr velocities", color='orange')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum.append([False, True, False, detection_delay, peak_excess, cumulative_excess, clean_corr_loss[-1], recovery_time])

        corr_clean_theta, corr_clean_gradients, corr_clean_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_clean", clean_theta, clean_velocities)
        corr_clean_loss = corr_clean_theta[:,0]**2 + corr_clean_theta[:,0]*corr_clean_theta[:,1] + corr_clean_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights clean velocities", color="green")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum.append([True, False, False, detection_delay, peak_excess, cumulative_excess, corr_clean_loss[-1], recovery_time])
        
        corr_corr_theta, corr_corr_gradients, corr_corr_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_corr", clean_theta, clean_velocities)
        corr_corr_loss = corr_corr_theta[:,0]**2 + corr_corr_theta[:,0]*corr_corr_theta[:,1] + corr_corr_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights corr velocities", color="blue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum.append([True, True, False, detection_delay, peak_excess, cumulative_excess, corr_corr_loss[-1], recovery_time])

        ax.grid()
        plt.legend()
        fig.savefig(f"{label}/2D_momentum/no_reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()


        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep", ylabel="Loss values", title=f"2D Momentum (f(x,y) = x^2 + xy + y^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_gradients, clean_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "clean", [], [])
        clean_loss = clean_theta[:,0]**2 + clean_theta[:,0]*clean_theta[:,1] + clean_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean velocities", color='black')
        momentum_2d_row_accum.append([False, False, False, detection_delay, 0, 0, clean_loss[-1], 0])

        clean_reset_theta, clean_reset_gradients, clean_reset_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "clean_reset", clean_theta, clean_velocities)
        clean_reset_loss = clean_reset_theta[:,0]**2 + clean_reset_theta[:,0]*clean_reset_theta[:,1] + clean_reset_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_reset_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights reset velocities", color='red')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum.append([False, False, True, detection_delay, peak_excess, cumulative_excess, clean_reset_loss[-1], recovery_time])

        corr_reset_theta, corr_reset_gradients, corr_reset_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, detection_delay, delta, "corr_reset", clean_theta, clean_velocities)  
        corr_reset_loss = corr_reset_theta[:,0]**2 + corr_reset_theta[:,0]*corr_reset_theta[:,1] + corr_reset_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_reset_loss[t_corruption:]-clean_loss[t_corruption:], label="Corrupted weights reset velocities", color='lightpink')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_loss, t_corruption, epsilon, detection_delay)
        momentum_2d_row_accum.append([True, False, True, detection_delay, peak_excess, cumulative_excess, corr_reset_loss[-1], recovery_time])

        ax.grid()
        plt.legend()
        fig.savefig(f"{label}/2D_momentum/reset/{detection_delay}_step_delay.png")
        # plt.show()
        plt.close()


        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep", ylabel="Loss values", title=f"2D Adam (f(x,y) = x^2+xy+y^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_m, clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean", [], [], [])
        clean_loss = clean_theta[:,0]**2 + clean_theta[:,0]*clean_theta[:,1] + clean_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean m clean v", color='black')
        adam_2d_row_accum.append([False, False, False, False, False, detection_delay, 0, 0, clean_loss[-1], 0])

        clean_clean_corr_theta, clean_clean_corr_m, clean_clean_corr_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_clean_corr", clean_theta, clean_m, clean_v)
        clean_clean_corr_loss = clean_clean_corr_theta[:,0]**2 + clean_clean_corr_theta[:,0]*clean_clean_corr_theta[:,1] + clean_clean_corr_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_clean_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean m corr v", color="orange")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_clean_corr_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([False, False, True, False, False, detection_delay, peak_excess, cumulative_excess, clean_clean_corr_loss[-1], recovery_time])

        clean_corr_clean_theta, clean_corr_clean_m, clean_corr_clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_corr_clean", clean_theta, clean_m, clean_v)
        clean_corr_clean_loss = clean_corr_clean_theta[:,0]**2 + clean_corr_clean_theta[:,0]*clean_corr_clean_theta[:,1] + clean_corr_clean_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_corr_clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights corr m clean v", color="olive")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_clean_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([False, True, False, False, False, detection_delay, peak_excess, cumulative_excess, clean_corr_clean_loss[-1], recovery_time])

        clean_corr_corr_theta, clean_corr_corr_m, clean_corr_corr_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_corr_corr", clean_theta, clean_m, clean_v)
        clean_corr_corr_loss = clean_corr_corr_theta[:,0]**2 + clean_corr_corr_theta[:,0]*clean_corr_corr_theta[:,1] + clean_corr_corr_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_corr_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights corr m corr v", color="green")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_corr_corr_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([False, True, True, False, False, detection_delay, peak_excess, cumulative_excess, clean_corr_corr_loss[-1], recovery_time])

        corr_clean_clean_theta, corr_clean_clean_m, corr_clean_clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_clean_clean", clean_theta, clean_m, clean_v)
        corr_clean_clean_loss = corr_clean_clean_theta[:,0]**2 + corr_clean_clean_theta[:,0]*corr_clean_clean_theta[:,1] + corr_clean_clean_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_clean_clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights clean m clean v", color="cyan")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_clean_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([True, False, False, False, False, detection_delay, peak_excess, cumulative_excess, corr_clean_clean_loss[-1], recovery_time])

        corr_clean_corr_theta, corr_clean_corr_m, corr_clean_corr_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_clean_corr", clean_theta, clean_m, clean_v)
        corr_clean_corr_loss = corr_clean_corr_theta[:,0]**2 + corr_clean_corr_theta[:,0]*corr_clean_corr_theta[:,1] + corr_clean_corr_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_clean_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights clean m corr v", color="blue")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_clean_corr_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([True, False, True, False, False, detection_delay, peak_excess, cumulative_excess, corr_clean_corr_loss[-1], recovery_time])

        corr_corr_clean_theta, corr_corr_clean_m, corr_corr_clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_corr_clean", clean_theta, clean_m, clean_v)
        corr_corr_clean_loss = corr_corr_clean_theta[:,0]**2 + corr_corr_clean_theta[:,0]*corr_corr_clean_theta[:,1] + corr_corr_clean_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_corr_clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights corr m clean v", color="purple")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_clean_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([True, True, False, False, False, detection_delay, peak_excess, cumulative_excess, corr_corr_clean_loss[-1], recovery_time])

        corr_corr_corr_theta, corr_corr_corr_m, corr_corr_corr_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_corr_corr", clean_theta, clean_m, clean_v)
        corr_corr_corr_loss = corr_corr_corr_theta[:,0]**2 + corr_corr_corr_theta[:,0]*corr_corr_corr_theta[:,1] + corr_corr_corr_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_corr_corr_loss[t_corruption:]-clean_loss[t_corruption:], label="Corr weights corr m corr v", color="pink")
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_corr_corr_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([True, True, True, False, False, detection_delay, peak_excess, cumulative_excess, corr_corr_corr_loss[-1], recovery_time])

        ax.grid()
        plt.legend()
        fig.savefig(f"{label}/2D_adam/no_reset/{detection_delay}_step_delay.png")
        plt.close()
        # plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlabel="Timestep", ylabel="Loss values", title=f"2D Adam (f(x,y) = x^2+xy+y^2) after {label} corruption (detection delay = {detection_delay})")

        clean_theta, clean_m, clean_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean", [], [], [])
        clean_loss = clean_theta[:,0]**2 + clean_theta[:,0]*clean_theta[:,1] + clean_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights clean m clean v", color='black')
        adam_2d_row_accum.append([False, False, False, False, False, detection_delay, 0, 0, clean_loss[-1], 0])

        clean_reset_moments_theta, clean_reset_moments_m, clean_reset_moments_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_reset_moments   ", clean_theta, clean_m, clean_v)
        clean_reset_moments_loss = clean_reset_moments_theta[:,0]**2 + clean_reset_moments_theta[:,0]*clean_reset_moments_theta[:,1] + clean_reset_moments_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_reset_moments_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights reset m reset v", color='red')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_moments_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([False, False, False, True, False, detection_delay, peak_excess, cumulative_excess, clean_reset_moments_loss[-1], recovery_time])

        clean_reset_optim_theta, clean_reset_optim_m, clean_reset_optim_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "clean_reset_optim", clean_theta, clean_m, clean_v)
        clean_reset_optim_loss = clean_reset_optim_theta[:,0]**2 + clean_reset_optim_theta[:,0]*clean_reset_optim_theta[:,1] + clean_reset_optim_theta[:, 1]**2
        ax.plot(t[t_corruption:], clean_reset_optim_loss[t_corruption:]-clean_loss[t_corruption:], label="Clean weights reset m reset v restart optim", color='maroon')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, clean_reset_optim_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([False, False, False, True, True, detection_delay, peak_excess, cumulative_excess, clean_reset_optim_loss[-1], recovery_time])

        corr_reset_moments_theta, corr_reset_m, corr_reset_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_reset_moments", clean_theta, clean_m, clean_v)
        corr_reset_loss = corr_reset_moments_theta[:,0]**2 + corr_reset_moments_theta[:,0]*corr_reset_moments_theta[:,1] + corr_reset_moments_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_reset_loss[t_corruption:]-clean_loss[t_corruption:], label="Corrupted weights reset m reset v", color='darkorange')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([True, False, False, True, False, detection_delay, peak_excess, cumulative_excess, corr_reset_loss[-1], recovery_time])

        corr_reset_optim_theta, corr_reset_optim_m, corr_reset_optim_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, detection_delay, delta, "corr_reset_optim", clean_theta, clean_m, clean_v)
        corr_reset_optim_loss = corr_reset_optim_theta[:,0]**2 + corr_reset_optim_theta[:,0]*corr_reset_optim_theta[:,1] + corr_reset_optim_theta[:, 1]**2
        ax.plot(t[t_corruption:], corr_reset_optim_loss[t_corruption:]-clean_loss[t_corruption:], label="Corrupted weights reset m reset v restart optim", color='lightpink')
        peak_excess, cumulative_excess, recovery_time = calculate_metrics(clean_loss, corr_reset_optim_loss, t_corruption, epsilon, detection_delay)
        adam_2d_row_accum.append([True, False, False, True, True, detection_delay, peak_excess, cumulative_excess, corr_reset_optim_loss[-1], recovery_time])

        ax.grid()
        plt.legend()
        fig.savefig(f"{label}/2D_adam/reset/{detection_delay}_step_delay.png")
        plt.close()
        # plt.show()


    momentum_1d_results = pd.DataFrame(momentum_1d_row_accum, columns=["corrupted weights?", "corrupted velocities?", "reset?", "detection delay", "peak excess loss", "cumulative excess loss", "final parameter value", "recovery time"])
    adam_1d_results = pd.DataFrame(adam_1d_row_accum, columns=["corrupted weights?", "corrupted m?", "corrupted v?", "reset?", "restarted optim?", "detection delay", "peak excess loss", "cumulative excess loss", "final parameter value", "recovery time"])
    momentum_2d_results = pd.DataFrame(momentum_2d_row_accum, columns=["corrupted weights?", "corrupted velocities?", "reset?", "detection delay", "peak excess loss", "cumulative excess loss", "final parameter value", "recovery time"])
    adam_2d_results = pd.DataFrame(adam_2d_row_accum, columns=["corrupted weights?", "corrupted m?", "corrupted v?", "reset?", "restarted optim?", "detection delay", "peak excess loss", "cumulative excess loss", "final parameter value", "recovery time"])

    momentum_1d_results.to_csv(f"{label}/results/1D_momentum_results.csv", index=False)
    adam_1d_results.to_csv(f"{label}/results/1D_adam_results.csv", index=False)
    momentum_2d_results.to_csv(f"{label}/results/2D_momentum_results.csv", index=False)
    adam_2d_results.to_csv(f"{label}/results/2D_adam_results.csv", index=False)