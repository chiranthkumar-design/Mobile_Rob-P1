"""
=============================================================================
CONTRIBUTION 2 — PARTICLE FILTER (PF) POSE SMOOTHER
Mobile Robotics M2 | University of Salford
Webots Simulation Controller

Description
-----------
This controller runs the same baseline odometry route (n0→n5) as the
mandatory assignment, but adds a Particle Filter smoother running in
parallel. At each waypoint the controller records:
  - Raw odometry pose   (from GPS — acting as ground truth comparison)
  - PF smoothed pose    (estimated from motion model + observation noise)
  - Destination error for both

The PF is a pose smoother (not a localiser): it uses wheel-velocity
predictions and a Gaussian observation model to reduce drift compared
with raw odometry, matching the ParticleFilterLocalizer in pioneer__2_.py.

Data logged to CSV (odom_pf_results.csv):
  run, waypoint, target_x, target_y,
  raw_x, raw_y, raw_err_m,
  pf_x,  pf_y,  pf_err_m,
  pf_improvement_%,
  time_s, n_particles

Statistical summary printed at the end (mean, std, % improvement).
=============================================================================
"""

from controller import Robot, Motor, GPS, Compass, InertialUnit
import math
import csv
import time
import random

# ─────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────
TIME_STEP       = 64            # ms
MAX_SPEED       = 6.28          # rad/s
WHEEL_RADIUS    = 0.0975        # m
AXLE_LENGTH     = 0.33          # m

GAMMA_THETA_DEG = 12.0
GAMMA_DIST_M    = 0.10          # waypoint acceptance radius

N_RUNS          = 5
N_PARTICLES     = 250
CSV_FILE        = "odom_pf_results.csv"

# ─────────────────────────────────────────────
# BASELINE WAYPOINTS (mm → m, same as pioneer__2_.py NODES_ODOM)
# Relative to robot start (0,0)
# ─────────────────────────────────────────────
WAYPOINTS_M = [
    (-0.510, +0.000),   # P1 — 510mm west  | Webots (1.327, 0.911)m
    (-0.510, -1.600),   # P2 — 1600mm south | Webots (1.327,-0.689)m
    (-0.510, -2.600),   # P3 — 2600mm south | Webots (1.327,-1.689)m
    (-1.510, -2.600),   # P4 — west+south   | Webots (0.327,-1.689)m
    (-0.510, -2.000),   # P5 — east+north   | Webots (1.327,-1.089)m — stays SOUTH of bed (bed south edge = -1.701m GPS-rel)
    # All 5 points A*-verified free inside Webots room bounds
]
POINT_NAMES = [f"P{i+1}" for i in range(len(WAYPOINTS_M))]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def deg360(a):
    return a % 360.0

def bearing_to_deg360(x0, y0, gx, gy):
    a = math.degrees(math.atan2(gy - y0, gx - x0))
    return a % 360.0

# ─────────────────────────────────────────────
# PARTICLE FILTER SMOOTHER
# (ported directly from pioneer__2_.py ParticleFilterLocalizer)
# ─────────────────────────────────────────────
class PFSmoother:
    """
    Particle Filter pose smoother.

    State per particle: [x_m, y_m, th_deg360]

    Prediction:
      - Apply motion model (v, w) with Gaussian noise
      - Matches the dynamics model in pioneer__2_.py

    Update (observation):
      - Observation = GPS pose (acts as noisy sensor reading)
      - Particles reweighted by Gaussian likelihood
      - Systematic resampling when Neff < 0.5*N

    Waypoint anchoring:
      - When the robot is within goal_gate_m of the current waypoint,
        an extra pseudo-measurement pulls the belief toward the waypoint
        (mirrors pioneer__2_.py goal_gate_mm / goal_sigma_xy)
    """

    def __init__(self, n=N_PARTICLES):
        self.n = n
        # Particles: list of [x, y, th_deg360]
        self.particles = [[0.0, 0.0, 0.0] for _ in range(n)]
        self.weights   = [1.0 / n] * n
        self.initialized = False
        self._rng = random.Random(42)   # reproducible

        # Noise parameters (m, deg) — tuned to match pioneer__2_.py
        self.init_sigma_xy    = 0.120   # wider init spread — robot may not be exactly at GPS reading
        self.init_sigma_th    = 10.0   # wider heading init
        self.proc_sigma_xy    = 0.018  # slightly higher process noise for robustness
        self.proc_sigma_th    = 2.5
        self.meas_sigma_xy    = 0.060  # more tolerant GPS measurement model
        self.meas_sigma_th    = 6.0    # more tolerant heading model

        # Waypoint anchor
        self.goal_gate_m      = 0.240
        self.goal_sigma_m     = 0.055
        self.goal_enabled     = True

    # ── Public interface ────────────────────────────────────────────────
    def reset(self, x, y, th_deg):
        """Reinitialise particles around (x, y, th_deg)."""
        for i in range(self.n):
            self.particles[i] = [
                x  + self._gauss(0.0, self.init_sigma_xy),
                y  + self._gauss(0.0, self.init_sigma_xy),
                deg360(th_deg + self._gauss(0.0, self.init_sigma_th))
            ]
        self.weights = [1.0 / self.n] * self.n
        self.initialized = True

    def step(self, x_obs, y_obs, th_obs_deg, v_ms, w_rads, dt, goal=None):
        """
        Run one PF step.  Returns (est_x, est_y, est_th_deg).

        x_obs, y_obs, th_obs_deg — GPS/compass observation (noisy ground truth)
        v_ms, w_rads             — wheel velocity commands sent this step
        dt                       — time delta (s)
        goal                     — (gx, gy) in metres, or None
        """
        if not self.initialized:
            self.reset(x_obs, y_obs, th_obs_deg)
            return x_obs, y_obs, th_obs_deg

        self._predict(v_ms, w_rads, dt)
        self._update(x_obs, y_obs, th_obs_deg, goal)
        return self._estimate()

    # ── Internal steps ──────────────────────────────────────────────────
    def _predict(self, v, w_rads, dt):
        dt = max(0.001, min(0.25, dt))
        for p in self.particles:
            v_noise = self._gauss(0.0, self.proc_sigma_xy + 0.03 * abs(v))
            w_noise = self._gauss(0.0,
                                  math.radians(self.proc_sigma_th)
                                  + 0.02 * abs(w_rads))
            th_rad = math.radians(p[2])
            v_eff  = v + v_noise
            w_eff  = w_rads + w_noise
            p[0]  += v_eff * dt * math.cos(th_rad)
            p[1]  += v_eff * dt * math.sin(th_rad)
            p[2]   = deg360(p[2] + math.degrees(w_eff * dt))

    def _update(self, x_obs, y_obs, th_obs_deg, goal):
        log_weights = []
        for p in self.particles:
            dx   = p[0] - x_obs
            dy   = p[1] - y_obs
            dth  = wrap_deg(p[2] - th_obs_deg)
            logw = (-0.5 * (dx  / self.meas_sigma_xy) ** 2
                    - 0.5 * (dy  / self.meas_sigma_xy) ** 2
                    - 0.5 * (dth / self.meas_sigma_th) ** 2)

            # Waypoint anchor
            if self.goal_enabled and goal is not None:
                gx, gy = goal
                raw_d = math.hypot(x_obs - gx, y_obs - gy)
                if raw_d <= self.goal_gate_m:
                    gdist = math.hypot(p[0] - gx, p[1] - gy)
                    logw += -0.5 * (gdist / self.goal_sigma_m) ** 2

            log_weights.append(logw)

        # Numerically stable weight normalisation
        max_lw = max(log_weights)
        ws = [math.exp(lw - max_lw) + 1e-300 for lw in log_weights]
        total = sum(ws)
        self.weights = [w / total for w in ws]

        # Effective sample size → resample if needed
        neff = 1.0 / sum(w * w for w in self.weights)
        if neff < 0.5 * self.n:
            self._resample()

    def _resample(self):
        """Systematic resampling."""
        cdf = []
        acc = 0.0
        for w in self.weights:
            acc += w
            cdf.append(acc)
        start = self._rng.random() / self.n
        points = [start + i / self.n for i in range(self.n)]
        new_particles = []
        j = 0
        for pt in points:
            while j < len(cdf) - 1 and cdf[j] < pt:
                j += 1
            src = self.particles[j]
            new_particles.append([src[0], src[1], src[2]])
        self.particles = new_particles
        self.weights = [1.0 / self.n] * self.n

    def _estimate(self):
        """Weighted mean position and circular mean heading."""
        x = sum(w * p[0] for w, p in zip(self.weights, self.particles))
        y = sum(w * p[1] for w, p in zip(self.weights, self.particles))
        # Circular mean for heading
        s = sum(w * math.sin(math.radians(p[2]))
                for w, p in zip(self.weights, self.particles))
        c = sum(w * math.cos(math.radians(p[2]))
                for w, p in zip(self.weights, self.particles))
        th = math.degrees(math.atan2(s, c)) % 360.0
        return x, y, th

    # ── Utilities ───────────────────────────────────────────────────────
    def _gauss(self, mu, sigma):
        return self._rng.gauss(mu, sigma)

# ─────────────────────────────────────────────
# ODOMETRY CONTROLLER (same as contribution1)
# ─────────────────────────────────────────────
class OdomController:
    def __init__(self):
        self.gamma_theta = GAMMA_THETA_DEG
        self.gamma_dist  = GAMMA_DIST_M
        self.kp_rot      = 1.3
        self.kp_steer    = 1.2
        self.v_cruise    = 0.25
        self.w_max       = math.radians(45.0)
        self.min_w       = math.radians(10.0)
        self._stage      = "ALIGN"

    def reset(self):
        self._stage = "ALIGN"

    def step(self, x, y, th_deg, gx, gy):
        dist = math.hypot(gx - x, gy - y)
        if dist <= self.gamma_dist:
            self._stage = "ALIGN"
            return 0.0, 0.0, "ARRIVED", True
        target = bearing_to_deg360(x, y, gx, gy)
        err    = wrap_deg(target - th_deg)
        if self._stage == "MOVE":
            if abs(err) > self.gamma_theta + 5.0:
                self._stage = "ALIGN"
        else:
            if abs(err) <= self.gamma_theta:
                self._stage = "MOVE"
        if self._stage == "ALIGN":
            w = clamp(math.radians(self.kp_rot * err), -self.w_max, self.w_max)
            if 0 < abs(w) < self.min_w:
                w = self.min_w if w > 0 else -self.min_w
            return 0.0, w, f"ALIGN err={err:.1f}", False
        w = clamp(math.radians(self.kp_steer * err), -self.w_max, self.w_max)
        return self.v_cruise, w, f"MOVE err={err:.1f}", False

# ─────────────────────────────────────────────
# DIFFERENTIAL DRIVE
# ─────────────────────────────────────────────
def drive(left_motor, right_motor, v_ms, w_rads):
    vl = (v_ms - w_rads * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    vr = (v_ms + w_rads * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
    left_motor.setVelocity(clamp(vl, -MAX_SPEED, MAX_SPEED))
    right_motor.setVelocity(clamp(vr, -MAX_SPEED, MAX_SPEED))

# ─────────────────────────────────────────────
# SENSOR HELPERS
# ─────────────────────────────────────────────
def get_gps_xy(gps):
    # Webots Z-up world: GPS returns [x_East, y_North, z_Height]
    # Use pos[0]=East, pos[1]=North (NOT pos[2] which is height)
    pos = gps.getValues()
    return pos[0], pos[1]

def get_heading_deg(imu):
    # InertialUnit in Z-up Webots: getRollPitchYaw()[2] = yaw around Z
    # This is standard math angle: East=0°, North=90° — same as contribution1
    rpy = imu.getRollPitchYaw()
    return math.degrees(rpy[2]) % 360.0

# ─────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────
def compute_stats(values):
    if not values:
        return 0.0, 0.0
    n   = len(values)
    mu  = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / max(1, n - 1)
    return mu, math.sqrt(var)

# ─────────────────────────────────────────────
# CSV LOGGER
# ─────────────────────────────────────────────
def init_csv():
    with open(CSV_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "waypoint", "target_x_m", "target_y_m",
                    "raw_x_m", "raw_y_m", "raw_err_m",
                    "pf_x_m",  "pf_y_m",  "pf_err_m",
                    "pf_improvement_pct", "time_s", "n_particles"])
    print(f"[CSV] Logging to {CSV_FILE}")

def log_row(row):
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([round(v, 4) if isinstance(v, float) else v for v in row])

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    robot = Robot()
    dt_s  = TIME_STEP / 1000.0

    # ── Motors ──────────────────────────────────────────────────────────
    left_motor  = robot.getDevice("left wheel")
    right_motor = robot.getDevice("right wheel")
    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    # ── Sensors ─────────────────────────────────────────────────────────
    gps = robot.getDevice("gps")
    gps.enable(TIME_STEP)
    imu = robot.getDevice("inertial unit")   # use InertialUnit for heading — confirmed working in contribution1
    imu.enable(TIME_STEP)

    # ── Warm-up: 25 steps so GPS/IMU settle (prevents NaN) ──────────────
    print("[INIT] Warming up sensors (25 steps)...")
    for _ in range(25):
        robot.step(TIME_STEP)
    x0, y0 = get_gps_xy(gps)
    th0    = get_heading_deg(imu)
    print(f"[ROBOT] Start pose: ({x0:.3f}, {y0:.3f}) m  th={th0:.1f}°")

    # Adjust waypoints to be relative to actual start position in Webots
    waypoints = [(wx + x0, wy + y0) for wx, wy in WAYPOINTS_M]
    print("[WAYPOINTS]")
    for i, (wx, wy) in enumerate(waypoints):
        print(f"  {POINT_NAMES[i]}: ({wx:.3f}, {wy:.3f}) m")

    # ── Init CSV & controllers ───────────────────────────────────────────
    init_csv()
    odom = OdomController()
    pf   = PFSmoother(n=N_PARTICLES)

    # Accumulators for final statistics
    all_raw_errors  = []
    all_pf_errors   = []

    # ── MULTIPLE RUNS ────────────────────────────────────────────────────
    for run in range(1, N_RUNS + 1):
        print(f"\n{'='*55}")
        print(f" RUN {run}/{N_RUNS}  |  PF particles={N_PARTICLES}")
        print(f"{'='*55}")

        # Read current GPS for PF initialisation only
        x_cur, y_cur = get_gps_xy(gps)
        th_cur       = get_heading_deg(imu)
        print(f"[RUN {run}] Current pose: ({x_cur:.3f},{y_cur:.3f})m  th={th_cur:.1f}°")

        # FIX 2: Always compute waypoints from ORIGINAL start (x0,y0) so
        # the robot travels the SAME physical route every run. Computing
        # from x_cur means P1 shifts to wherever the robot ended up last
        # run — robot arrives in 0.1s and PF has no time to converge.
        waypoints = [(wx + x0, wy + y0) for wx, wy in WAYPOINTS_M]

        odom.reset()
        # FIX 1+3: reset PF to current actual position (not x0,y0).
        # Then set warmup counter: suppress goal anchor for first 20 steps
        # to prevent immediate weight collapse if robot starts near P1.
        pf.reset(x_cur, y_cur, th_cur)
        pf_warmup_steps = 20   # PF steps before goal anchoring is enabled

        wp_idx = 0
        t_run_start = time.time()
        # FIX 1: Use fixed simulation dt = TIME_STEP ms, NOT wall-clock time.
        # At 648x Webots speed wall-clock dt ≈ 0.0001s → particles barely
        # move → all weights collapse → PF diverges every run after run 1.
        dt_fixed = TIME_STEP / 1000.0   # always 0.064 s per step
        last_v, last_w = 0.0, 0.0

        run_raw_errors = []
        run_pf_errors  = []

        while robot.step(TIME_STEP) != -1:
            # ── Sensor readings ─────────────────────────────────────────
            x_raw, y_raw = get_gps_xy(gps)
            th_raw       = get_heading_deg(imu)

            # FIX 3: count down warmup — goal anchor suppressed until warm
            if pf_warmup_steps > 0:
                pf_warmup_steps -= 1
                goal_for_pf = None          # no anchor during warmup
            else:
                goal_for_pf = waypoints[wp_idx] if wp_idx < len(waypoints) else None

            # FIX 1: use dt_fixed (0.064s) — NOT wall-clock time
            pf_x, pf_y, pf_th = pf.step(
                x_raw, y_raw, th_raw,
                last_v, last_w, dt_fixed,   # ← was: dt (wall-clock, broken at high speed)
                goal=goal_for_pf
            )

            if wp_idx >= len(waypoints):
                # Route complete — stop
                drive(left_motor, right_motor, 0.0, 0.0)
                elapsed = time.time() - t_run_start
                print(f"[RUN {run}] ALL WAYPOINTS REACHED  t={elapsed:.1f}s")
                break

            gx, gy = waypoints[wp_idx]

            # ── Odometry controller ──────────────────────────────────────
            v, w, info, arrived = odom.step(x_raw, y_raw, th_raw, gx, gy)
            drive(left_motor, right_motor, v, w)
            last_v, last_w = v, w

            if arrived:
                # FIX 4: stop motors first so PF gets a clean stopped-state
                # GPS fix, not a mid-motion reading with stale (v,w) inputs
                drive(left_motor, right_motor, 0.0, 0.0)
                # One extra step so GPS settles after stopping
                robot.step(TIME_STEP)
                x_raw, y_raw = get_gps_xy(gps)
                th_raw       = get_heading_deg(imu)
                # Update PF with stopped state before recording error
                pf_x, pf_y, pf_th = pf.step(
                    x_raw, y_raw, th_raw,
                    0.0, 0.0, dt_fixed,
                    goal=(gx, gy)
                )

                raw_err = math.hypot(gx - x_raw, gy - y_raw)
                pf_err  = math.hypot(gx - pf_x,  gy - pf_y)
                elapsed = time.time() - t_run_start
                improv  = ((raw_err - pf_err) / raw_err * 100.0
                           if raw_err > 1e-6 else 0.0)

                print(f"[RUN {run}] {POINT_NAMES[wp_idx]:3s} | "
                      f"raw_err={raw_err*1000:.0f}mm  "
                      f"pf_err={pf_err*1000:.0f}mm  "
                      f"improv={improv:+.1f}%  t={elapsed:.1f}s")

                log_row([run, POINT_NAMES[wp_idx],
                         gx, gy,
                         x_raw, y_raw, raw_err,
                         pf_x,  pf_y,  pf_err,
                         improv, elapsed, N_PARTICLES])

                run_raw_errors.append(raw_err)
                run_pf_errors.append(pf_err)
                wp_idx += 1
                odom.reset()
                last_v, last_w = 0.0, 0.0  # reset velocity after waypoint

        # Per-run summary
        if run_raw_errors:
            mu_raw, sd_raw = compute_stats(run_raw_errors)
            mu_pf,  sd_pf  = compute_stats(run_pf_errors)
            ovr_improv = ((mu_raw - mu_pf) / mu_raw * 100.0
                          if mu_raw > 1e-6 else 0.0)
            print(f"  → Raw  mean={mu_raw*1000:.1f}mm  std={sd_raw*1000:.1f}mm")
            print(f"  → PF   mean={mu_pf*1000:.1f}mm   std={sd_pf*1000:.1f}mm")
            print(f"  → PF improvement over raw: {ovr_improv:+.1f}%")
            all_raw_errors.extend(run_raw_errors)
            all_pf_errors.extend(run_pf_errors)

        # Pause between runs (let Webots reset robot if needed)
        drive(left_motor, right_motor, 0.0, 0.0)
        for _ in range(int(2000 / TIME_STEP)):
            if robot.step(TIME_STEP) == -1:
                break

    # ── FINAL STATISTICS (all runs combined) ────────────────────────────
    print("\n" + "=" * 55)
    print(" FINAL STATISTICS (all runs)")
    print("=" * 55)
    if all_raw_errors:
        mu_raw, sd_raw = compute_stats(all_raw_errors)
        mu_pf,  sd_pf  = compute_stats(all_pf_errors)
        cov_raw = sd_raw / mu_raw if mu_raw > 1e-6 else 0.0
        cov_pf  = sd_pf  / mu_pf  if mu_pf  > 1e-6 else 0.0
        improv  = (mu_raw - mu_pf) / mu_raw * 100.0 if mu_raw > 1e-6 else 0.0

        # Correlation (raw error vs run index as proxy for drift)
        n = len(all_raw_errors)
        if n > 1:
            xs = list(range(1, n + 1))
            xm = sum(xs) / n
            ym_raw = mu_raw
            ym_pf  = mu_pf
            num_raw = sum((xs[i]-xm)*(all_raw_errors[i]-ym_raw) for i in range(n))
            num_pf  = sum((xs[i]-xm)*(all_pf_errors[i]-ym_pf) for i in range(n))
            den_x   = math.sqrt(sum((xi-xm)**2 for xi in xs))
            den_raw = math.sqrt(sum((v-ym_raw)**2 for v in all_raw_errors))
            den_pf  = math.sqrt(sum((v-ym_pf)**2 for v in all_pf_errors))
            rho_raw = num_raw/(den_x*den_raw) if den_x*den_raw>1e-12 else 0.0
            rho_pf  = num_pf/(den_x*den_pf)  if den_x*den_pf >1e-12 else 0.0
        else:
            rho_raw = rho_pf = 0.0

        print(f"  Metric              Raw Odometry       PF Smoother")
        print(f"  {'─'*48}")
        print(f"  Mean error        {mu_raw*1000:8.2f} mm       {mu_pf*1000:8.2f} mm")
        print(f"  Std deviation     {sd_raw*1000:8.2f} mm       {sd_pf*1000:8.2f} mm")
        print(f"  CoV (σ/μ)         {cov_raw:8.4f}           {cov_pf:8.4f}")
        print(f"  Correlation ρ     {rho_raw:8.4f}           {rho_pf:8.4f}")
        print(f"  PF improvement over raw: {improv:+.2f}%")
        print(f"  N particles: {N_PARTICLES}")
        print(f"  Total samples: {n}")
    print(f"\n[DONE] Results saved to {CSV_FILE}")
    print("[DONE] PF contribution complete.")

if __name__ == "__main__":
    main()