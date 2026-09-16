import numpy as np
import matplotlib.pyplot as plt

# ---------------- FILE PATH ----------------
file_path = r"D:\SALFORD\my_project\controllers\contribution2_pf\odom_pf_results.csv"

# ---------------- STORAGE ----------------
x_odom = []
y_odom = []
x_pf = []
y_pf = []

# ---------------- READ CSV ----------------
with open(file_path, "r") as f:
    for line in f:
        try:
            values = line.strip().split(",")

            # Extract correct columns:
            # raw_x_m = index 4
            # raw_y_m = index 5
            # pf_x_m  = index 7
            # pf_y_m  = index 8

            raw_x = float(values[4])
            raw_y = float(values[5])
            pf_x = float(values[7])
            pf_y = float(values[8])

            x_odom.append(raw_x)
            y_odom.append(raw_y)
            x_pf.append(pf_x)
            y_pf.append(pf_y)

        except:
            continue  # skip header or invalid rows

# Convert to numpy
x_odom = np.array(x_odom)
y_odom = np.array(y_odom)
x_pf = np.array(x_pf)
y_pf = np.array(y_pf)

# ---------------- ERROR ----------------
error = np.sqrt((x_odom - x_pf)**2 + (y_odom - y_pf)**2)

# ---------------- STATISTICS ----------------
print("\n===== PF vs ODOM STATISTICS =====")
print(f"Mean Error: {np.mean(error):.3f} m")
print(f"Std Dev   : {np.std(error):.3f} m")
print(f"Max Error : {np.max(error):.3f} m")

# ---------------- TRAJECTORY PLOT ----------------
plt.figure()

plt.plot(x_odom, y_odom, label="Odometry Path", linewidth=2)
plt.plot(x_pf, y_pf, linestyle='--', label="PF Path")

# Start & End
plt.scatter(x_odom[0], y_odom[0], label="Start")
plt.scatter(x_odom[-1], y_odom[-1], label="End")

# Kitchen Goal
plt.scatter(2.5, 2.5, marker='*', label="Goal (Kitchen)")

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Trajectory Comparison: Odometry vs Particle Filter")

plt.legend()
plt.grid()
plt.axis("equal")

plt.savefig("trajectory_plot.png", dpi=300)

# ---------------- ERROR PLOT ----------------
plt.figure()

plt.plot(error)
plt.title("Particle Filter Error Over Time")
plt.xlabel("Time Step")
plt.ylabel("Error (m)")
plt.grid()

plt.savefig("error_plot.png", dpi=300)

# ---------------- SHOW ----------------
plt.show()