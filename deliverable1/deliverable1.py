import math
import matplotlib.pyplot as plt
import numpy as np

"""
Variable initialization
"""
x_0 = 5
y_0 = 5
alpha = 0.1
gamma = 0.9
beta_1 = 0.9
beta_2 = 0.99
delta = 100
t_corruption = 50
num_timesteps = 100

t = np.arange(0, num_timesteps+1, 1)

"""
Functions
"""
def momentum_1d(x_0, alpha, gamma, t_corrupt, delta, strategy, ref_velocities):
    x = [x_0]
    gradients = [0]
    velocities = [0]
    for t in range(num_timesteps):
        gradient = 2 * x[-1]
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset":
            gradient += delta
        prev_velocity = velocities[-1]
        if t == t_corrupt + 1:
            if strategy == "reset" or strategy == "clean_reset":
                prev_velocity = gamma * 0 + alpha * gradient
            elif strategy == "replace":
                prev_velocity = ref_velocities[t]
        velocity = gamma * prev_velocity + alpha * gradient
        x.append(x[-1] - velocity)
        gradients.append(gradient)
        velocities.append(velocity)
    return np.array(x), gradients, velocities

def momentum_2d(x_0, y_0, alpha, gamma, t_corrupt, delta, strategy, ref_velocities):
    x = [np.array([x_0, y_0])]
    gradients = [np.array([0,0])]
    velocities = [np.array([0,0])]
    for t in range(num_timesteps):
        gradient = np.array([2*x[-1][0] + x[-1][1], x[-1][0] + 2*x[-1][1]])
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset":
            gradient += np.array([delta, 0])
        prev_velocity = velocities[-1]
        if t == t_corrupt + 1:
            if strategy == "reset" or strategy == "clean_reset":
                prev_velocity = gamma * 0 + alpha * gradient
            elif strategy == "replace":
                prev_velocity = ref_velocities[t]
        velocity = gamma * prev_velocity + alpha * gradient
        x.append(x[-1] - velocity)
        gradients.append(gradient)
        velocities.append(velocity)
    return np.array(x), np.array(gradients), np.array(velocities)

def adam_1d(x_0, alpha, beta_1, beta_2, t_corrupt, delta, strategy, ref_m, ref_v):
    x = [x_0]
    m_values = [0]
    v_values = [0]
    for t in range(num_timesteps):
        gradient = 2 * x[-1]
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset":
            gradient += delta
        prev_m = m_values[-1]
        prev_v = v_values[-1]
        if t == t_corrupt + 1:
            if strategy == "reset" or strategy == "clean_reset":
                prev_m = 0
                prev_v = 0
            elif strategy == "replace":
                prev_m = ref_m[t]
                prev_v = ref_v[t]
        m = beta_1 * prev_m + (1-beta_1)*gradient
        v = beta_2 * prev_v + (1-beta_2)*gradient**2
        hat_m = (m)/(1-beta_1**(t+1))
        hat_v = (v)/(1-beta_2**(t+1))
        x.append(x[-1] - ((alpha)/(math.sqrt(hat_v) + 10**(-8)))*hat_m)
        m_values.append(m)
        v_values.append(v)
    return np.array(x), np.array(m_values), np.array(v_values)

def adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corrupt, delta, strategy, ref_m, ref_v):
    theta = [np.array([x_0, y_0])]
    m_values = [np.array([0,0])]
    v_values = [np.array([0,0])]
    for t in range(num_timesteps):
        gradient = np.array([2*theta[-1][0] + theta[-1][1], theta[-1][0] + 2*theta[-1][1]])
        if t == t_corrupt and strategy != "clean" and strategy != "clean_reset":
            gradient += np.array([delta, 0])
        prev_m = m_values[-1]
        prev_v = v_values[-1]
        if t == t_corrupt + 1:
            if strategy == "reset" or strategy == "clean_reset":
                prev_m = 0
                prev_v = 0
            elif strategy == "replace":
                prev_m = ref_m[t]
                prev_v = ref_v[t]
        m = beta_1 * prev_m + (1-beta_1)*gradient
        v = beta_2 * prev_v + (1-beta_2)*gradient**2
        hat_m = (m)/(1-beta_1**(t+1))
        hat_v = (v)/(1-beta_2**(t+1))
        theta.append(theta[-1] - ((alpha)/(np.sqrt(hat_v) + 10**(-8)))*hat_m)
        m_values.append(m)
        v_values.append(v)
    return np.array(theta), np.array(m_values), np.array(v_values)

"""
1D experiments
"""

fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Parameter values", title=f"1D Momentum (y=x^2)")

# 1d example (y = x^2) with momentum - no corruption
uncorrupted_theta, uncorrupted_gradients, ref_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, delta, "clean", [])
ax.plot(t, uncorrupted_theta, label="Uncorrupted", color='red')

clean_reset_theta, clean_reset_gradients, clean_reset_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, delta, "clean_reset", [])
ax.plot(t, clean_reset_theta, label="Uncorrupted reset", color='orange')

# 1d example (y = x^2) with momentum - corruption
# keep
keep_theta, keep_gradients, keep_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, delta, "keep", ref_velocities)
ax.plot(t, keep_theta, label="Corrupted - keep", color="green")

# reset
reset_theta, reset_gradients, reset_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, delta, "reset", ref_velocities)
ax.plot(t, reset_theta, label="Corrupted - reset", color="blue")

# replace
replace_theta, replace_gradients, replace_velocities = momentum_1d(x_0, alpha, gamma, t_corruption, delta, "replace", ref_velocities)
ax.plot(t, replace_theta, label="Corrupted - replace", color="purple")

plt.axvline(x=t_corruption, color="black", linestyle="--")

ax.grid()
plt.legend()
fig.savefig(f"1D_momentum_params.png")
# plt.show()

# plotting deviation
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Deviation from uncorrupted trajectory", title="1D momentum deviation")

ax.plot(t, np.abs(clean_reset_theta-uncorrupted_theta), label="Uncorrupted reset", color="orange")
ax.plot(t, np.abs(keep_theta-uncorrupted_theta), label="Keep", color="green")
ax.plot(t, np.abs(reset_theta-uncorrupted_theta), label="Reset", color="blue")
ax.plot(t, np.abs(replace_theta-uncorrupted_theta), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"1D_momentum_delta.png")
# plt.show()

# plotting how much parameters change per timestep
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Parameter update", title="1D Momentum Parameter Updates(y=x^2)")

uncorrupted_diff = np.diff(uncorrupted_theta)
clean_reset_diff = np.diff(clean_reset_theta)
keep_diff = np.diff(keep_theta)
reset_diff = np.diff(reset_theta)
replace_diff = np.diff(replace_theta)
ax.plot(t[1:], uncorrupted_diff, label="Uncorrupted", color="red")
ax.plot(t[1:], clean_reset_diff, label="Uncorrupted reset", color="orange")
ax.plot(t[1:], keep_diff, label="Keep", color="green")
ax.plot(t[1:], reset_diff, label="Reset", color="blue")
ax.plot(t[1:], replace_diff, label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"1D_momentum_magnitude.png")
# plt.show()

# velocity plot
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Velocity", title="1D Momentum Velocity (y=x^2)")

ax.plot(t, ref_velocities, label="Uncorrupted", color="red")
ax.plot(t, clean_reset_velocities, label="Uncorrupted reset", color="orange")
ax.plot(t, keep_velocities, label="Keep", color="green")
ax.plot(t, reset_velocities, label="Reset", color="blue")
ax.plot(t, replace_velocities, label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"1D_momentum_velocity.png")
# plt.show()

fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Parameter values", title="1D Adam (y=x^2)")

# 1d example (y = x^2) with momentum - no corruption
uncorrupted_theta, ref_m, ref_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, delta, "clean", [], [])
ax.plot(t, uncorrupted_theta, label="Uncorrupted", color='red')

clean_reset_theta, clean_reset_m, clean_reset_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, delta, "clean_reset", [], [])
ax.plot(t, clean_reset_theta, label="Uncorrupted reset", color='orange')

# 1d example (y = x^2) with momentum - corruption
# keep
keep_theta, keep_m, keep_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, delta, "keep", ref_m, ref_v)
ax.plot(t, keep_theta, label="Corrupted - keep", color="green")

# reset
reset_theta, reset_m, reset_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, delta, "reset", ref_m, ref_v)
ax.plot(t, reset_theta, label="Corrupted - reset", color="blue")

# replace
replace_theta, replace_m, replace_v = adam_1d(x_0, alpha, beta_1, beta_2, t_corruption, delta, "replace", ref_m, ref_v)
ax.plot(t, replace_theta, label="Corrupted - replace", color="purple")

plt.axvline(x=t_corruption, color="black", linestyle="--")

ax.grid()
plt.legend()
fig.savefig(f"1D_adam_params.png")
# plt.show()

# plotting deviation
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Deviation from uncorrupted trajectory", title="1D Adam deviation")

ax.plot(t, np.abs(clean_reset_theta-uncorrupted_theta), label="Uncorrupted reset", color="orange")
ax.plot(t, np.abs(keep_theta-uncorrupted_theta), label="Keep", color="green")
ax.plot(t, np.abs(reset_theta-uncorrupted_theta), label="Reset", color="blue")
ax.plot(t, np.abs(replace_theta-uncorrupted_theta), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"1D_adam_delta.png")
# plt.show()

# plotting how much parameters change per timestep
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Parameter update", title="1D Adam Parameter Updates(y=x^2)")

uncorrupted_diff = np.diff(uncorrupted_theta)
clean_reset_diff = np.diff(clean_reset_theta)
keep_diff = np.diff(keep_theta)
reset_diff = np.diff(reset_theta)
replace_diff = np.diff(replace_theta)
ax.plot(t[1:], uncorrupted_diff, label="Uncorrupted", color="red")
ax.plot(t[1:], clean_reset_diff, label="Uncorrupted reset", color="orange")
ax.plot(t[1:], keep_diff, label="Keep", color="green")
ax.plot(t[1:], reset_diff, label="Reset", color="blue")
ax.plot(t[1:], replace_diff, label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"1D_adam_magnitude.png")
# plt.show()

# plotting first and second moment
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="First moment (m)", title="1D Adam First Moment (y=x^2)")

ax.plot(t, ref_m, label="Uncorrupted", color="red")
ax.plot(t, clean_reset_m, label="Uncorrupted reset", color="orange")
ax.plot(t, keep_m, label="Keep", color="green")
ax.plot(t, reset_m, label="Reset", color="blue")
ax.plot(t, replace_m, label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"1D_adam_m.png")
# plt.show()

fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Second moment (v)", title="1D Adam Second Moment (y=x^2)")

ax.plot(t, ref_v, label="Uncorrupted", color="red")
ax.plot(t, clean_reset_v, label="Uncorrupted reset", color="orange")
ax.plot(t, keep_v, label="Keep", color="green")
ax.plot(t, reset_v, label="Reset", color="blue")
ax.plot(t, replace_v, label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"1D_adam_v.png")
# plt.show()

"""
2D experiments
"""

fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Parameter values", title=f"2D Momentum (f(x,y) = x^2 + xy + y^2)")

# 2d example (f(x, y) = x^2 + xy + y^2) with momentum - no corruption
uncorrupted_theta, uncorrupted_gradients, ref_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, delta, "clean", [])
ax.plot(t, np.sqrt(uncorrupted_theta[:, 0]**2 + uncorrupted_theta[:, 1]**2), label="Uncorrupted", color='red')

clean_reset_theta, clean_reset_gradients, clean_reset_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, delta, "clean_reset", [])
ax.plot(t, np.sqrt(clean_reset_theta[:, 0]**2 + clean_reset_theta[:, 1]**2), label="Uncorrupted reset", color='orange')

# 2d example (f(x, y) = x^2 + xy + y^2) with momentum - corruption
# keep
keep_theta, keep_gradients, keep_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, delta, "keep", ref_velocities)
ax.plot(t, np.sqrt(keep_theta[:, 0]**2 + keep_theta[:, 1]**2), label="Corrupted - keep", color="green")

# reset
reset_theta, reset_gradients, reset_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, delta, "reset", ref_velocities)
ax.plot(t, np.sqrt(reset_theta[:, 0]**2 + reset_theta[:, 1]**2), label="Corrupted - reset", color="blue")

# replace
replace_theta, replace_gradients, replace_velocities = momentum_2d(x_0, y_0, alpha, gamma, t_corruption, delta, "replace", ref_velocities)
ax.plot(t, np.sqrt(replace_theta[:, 0]**2 + replace_theta[:, 1]**2), label="Corrupted - replace", color="purple")

plt.axvline(x=t_corruption, color="black", linestyle="--")

ax.grid()
plt.legend()
fig.savefig(f"2D_momentum_params.png")
# plt.show()

# plotting deviation
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Deviation from uncorrupted trajectory", title="2D Momentum deviation")

ax.plot(t, np.sqrt((clean_reset_theta[:, 0]-uncorrupted_theta[:, 0])**2 + (clean_reset_theta[:, 1]-uncorrupted_theta[:, 1])**2), label="Uncorrupted reset", color="orange")
ax.plot(t, np.sqrt((keep_theta[:, 0]-uncorrupted_theta[:, 0])**2 + (keep_theta[:, 1]-uncorrupted_theta[:, 1])**2), label="Keep", color="green")
ax.plot(t, np.sqrt((reset_theta[:, 0]-uncorrupted_theta[:, 0])**2 + (reset_theta[:, 1]-uncorrupted_theta[:, 1])**2), label="Reset", color="blue")
ax.plot(t, np.sqrt((replace_theta[:, 0]-uncorrupted_theta[:, 0])**2 + (replace_theta[:, 1]-uncorrupted_theta[:, 1])**2), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"2D_momentum_delta.png")
# plt.show()

# plotting how much parameters change per timestep
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Parameter update", title="2D Momentum parameter updates")

uncorrupted_diff = np.diff(uncorrupted_theta, axis=0)
clean_reset_diff = np.diff(clean_reset_theta, axis=0)
keep_diff = np.diff(keep_theta, axis=0)
reset_diff = np.diff(reset_theta, axis=0)
replace_diff = np.diff(replace_theta, axis=0)
ax.plot(t[1:], np.sqrt(uncorrupted_diff[:, 0]**2 + uncorrupted_diff[:, 1]**2), label="Uncorrupted", color='red')
ax.plot(t[1:], np.sqrt(clean_reset_diff[:, 0]**2 + clean_reset_diff[:, 1]**2), label="Uncorrupted reset", color='orange')
ax.plot(t[1:], np.sqrt(keep_diff[:, 0]**2 + keep_diff[:, 1]**2), label="Keep", color="green")
ax.plot(t[1:], np.sqrt(reset_diff[:, 0]**2 + reset_diff[:, 1]**2), label="Reset", color="blue")
ax.plot(t[1:], np.sqrt(replace_diff[:, 0]**2 + replace_diff[:, 1]**2), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"2D_momentum_magnitude.png")
# plt.show()

# velocity plot
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Velocity", title="2D Momentum Velocity (f(x,y) = x^2+xy+y^2)")

ax.plot(t, np.sqrt(ref_velocities[:, 0]**2 + ref_velocities[:, 1]**2), label="Uncorrupted", color="red")
ax.plot(t, np.sqrt(clean_reset_velocities[:, 0]**2 + clean_reset_velocities[:, 1]**2), label="Uncorrupted reset", color="orange")
ax.plot(t, np.sqrt(keep_velocities[:, 0]**2 + keep_velocities[:, 1]**2), label="Keep", color="green")
ax.plot(t, np.sqrt(reset_velocities[:, 0]**2 + reset_velocities[:, 1]**2), label="Reset", color="blue")
ax.plot(t, np.sqrt(replace_velocities[:, 0]**2 + replace_velocities[:, 1]**2), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"2D_momentum_velocity.png")
# plt.show()

fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Parameter values", title="2D Adam (f(x,y) = x^2+xy+y^2)")

# 2d Adam example (f(x, y) = x^2 + xy + y^2) - no corruption
uncorrupted_theta, ref_m, ref_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, delta, "clean", [], [])
ax.plot(t, np.sqrt(uncorrupted_theta[:, 0]**2 + uncorrupted_theta[:, 1]**2), label="Uncorrupted", color='red')

clean_reset_theta, clean_reset_m, clean_reset_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, delta, "clean_reset", [], [])
ax.plot(t, np.sqrt(clean_reset_theta[:, 0]**2 + clean_reset_theta[:, 1]**2), label="Uncorrupted reset", color='orange')

# 2d example (f(x, y) = x^2 + xy + y^2) with Adam - corruption
# keep
keep_theta, keep_m, keep_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, delta, "keep", ref_m, ref_v)
ax.plot(t, np.sqrt(keep_theta[:, 0]**2 + keep_theta[:, 1]**2), label="Corrupted - keep", color="green")

# reset
reset_theta, reset_m, reset_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, delta, "reset", ref_m, ref_v)
ax.plot(t, np.sqrt(reset_theta[:, 0]**2 + reset_theta[:, 1]**2), label="Corrupted - reset", color="blue")

# replace
replace_theta, replace_m, replace_v = adam_2d(x_0, y_0, alpha, beta_1, beta_2, t_corruption, delta, "replace", ref_m, ref_v)
ax.plot(t, np.sqrt(replace_theta[:, 0]**2 + replace_theta[:, 1]**2), label="Corrupted - replace", color="purple")

plt.axvline(x=t_corruption, color="black", linestyle="--")

ax.grid()
plt.legend()
fig.savefig(f"2D_adam_params.png")
# plt.show()

# plotting deviation
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Deviation from uncorrupted trajectory", title="2D Adam deviation")

ax.plot(t, np.sqrt((clean_reset_theta[:, 0]-uncorrupted_theta[:, 0])**2 + (clean_reset_theta[:, 1]-uncorrupted_theta[:, 1])**2), label="Uncorrupted reset", color="orange")
ax.plot(t, np.sqrt((keep_theta[:, 0]-uncorrupted_theta[:, 0])**2 + (keep_theta[:, 1]-uncorrupted_theta[:, 1])**2), label="Keep", color="green")
ax.plot(t, np.sqrt((reset_theta[:, 0]-uncorrupted_theta[:, 0])**2 + (reset_theta[:, 1]-uncorrupted_theta[:, 1])**2), label="Reset", color="blue")
ax.plot(t, np.sqrt((replace_theta[:, 0]-uncorrupted_theta[:, 0])**2 + (replace_theta[:, 1]-uncorrupted_theta[:, 1])**2), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"2D_adam_delta.png")
# plt.show()

# plotting how much parameters change per timestep
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Parameter update", title="2D Adam parameter updates")

uncorrupted_diff = np.diff(uncorrupted_theta, axis=0)
clean_reset_diff = np.diff(clean_reset_theta, axis=0)
keep_diff = np.diff(keep_theta, axis=0)
reset_diff = np.diff(reset_theta, axis=0)
replace_diff = np.diff(replace_theta, axis=0)
ax.plot(t[1:], np.sqrt(uncorrupted_diff[:, 0]**2 + uncorrupted_diff[:, 1]**2), label="Uncorrupted", color='red')
ax.plot(t[1:], np.sqrt(clean_reset_diff[:, 0]**2 + clean_reset_diff[:, 1]**2), label="Uncorrupted reset", color='orange')
ax.plot(t[1:], np.sqrt(keep_diff[:, 0]**2 + keep_diff[:, 1]**2), label="Keep", color="green")
ax.plot(t[1:], np.sqrt(reset_diff[:, 0]**2 + reset_diff[:, 1]**2), label="Reset", color="blue")
ax.plot(t[1:], np.sqrt(replace_diff[:, 0]**2 + replace_diff[:, 1]**2), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"2D_adam_magnitude.png")
# plt.show()

# plotting first and second moment
fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="First moment (m)", title="2D Adam First Moment (f(x,y) = x^2+xy+y^2)")

ax.plot(t, np.sqrt(ref_m[:, 0]**2 + ref_m[:, 1]**2), label="Uncorrupted", color="red")
ax.plot(t, np.sqrt(clean_reset_m[:, 0]**2 + clean_reset_m[:, 1]**2), label="Uncorrupted reset", color="orange")
ax.plot(t, np.sqrt(keep_m[:, 0]**2 + keep_m[:, 1]**2), label="Keep", color="green")
ax.plot(t, np.sqrt(reset_m[:, 0]**2 + reset_m[:, 1]**2), label="Reset", color="blue")
ax.plot(t, np.sqrt(replace_m[:, 0]**2 + replace_m[:, 1]**2), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"2D_adam_m.png")
# plt.show()

fig, ax = plt.subplots()
ax.set(xlabel="Timestep", ylabel="Second moment (v)", title="2D Adam Second Moment (f(x,y) = x^2+xy+y^2)")

ax.plot(t, np.sqrt(ref_v[:, 0]**2 + ref_v[:, 1]**2), label="Uncorrupted", color="red")
ax.plot(t, np.sqrt(clean_reset_v[:, 0]**2 + clean_reset_v[:, 1]**2), label="Uncorrupted reset", color="orange")
ax.plot(t, np.sqrt(keep_v[:, 0]**2 + keep_v[:, 1]**2), label="Keep", color="green")
ax.plot(t, np.sqrt(reset_v[:, 0]**2 + reset_v[:, 1]**2), label="Reset", color="blue")
ax.plot(t, np.sqrt(replace_v[:, 0]**2 + replace_v[:, 1]**2), label="Replace", color="purple")

ax.grid()
plt.legend()
fig.savefig(f"2D_adam_v.png")
# plt.show()