"""
GOAL: Webots (1.93, 3.57, 0) m → GPS-relative (93, 2659) mm
ROUTE: south → west → north → east (around kitchen walls)
=============================================================================
"""
from controller import Robot
import math, heapq, csv, os, time

# CONFIRMED DEVICE NAMES  (from .wbt file )

MOTOR_L  = "left wheel"
MOTOR_R  = "right wheel"
ENC_L    = "left wheel sensor"
ENC_R    = "right wheel sensor"
GPS_N    = "gps"
IMU_N    = "inertial unit"
LIDAR_N  = "Sick LMS 291"
SONARS   = [f"so{i}" for i in range(16)]

# ROBOT PHYSICAL CONSTANTS  (Pioneer P3-DX)

TIME_STEP  = 64           # ms — must match Webots basicTimeStep
WHEEL_R    = 0.0975       # m  wheel radius
AXLE       = 0.330        # m  wheel track
MAX_WHEEL  = 6.28         # rad/s motor limit
MAX_W      = math.radians(60.0)


# GOAL  (GPS-relative mm, robot start = 0,0)
# Webots world (1.93, 3.57, 0) m
# GPS-relative: (1.93-1.8373)*1000=93  (3.57-0.9109)*1000=2659

GOAL_X_MM  =   93.0
GOAL_Y_MM  = 2659.0


# MAP PARAMETERS  (GPS-relative mm, confirmed from .wbt)

MAP_RES_MM  = 50          # grid cell size mm
INFLATE_MM  = 220         # obstacle inflation mm (robot clearance)
MAP_X1 = -4360.0;  MAP_X2 =  690.0
MAP_Y1 = -5670.0;  MAP_Y2 = 3850.0

# Wall bounding boxes (GPS-relative mm, from .wbt Wall proto positions)
WALL_BOXES = [
    (  473, -5661,    673,  3839),   # Wall R  (east boundary)
    (-4327, -5671,  -4127,  3829),   # Wall L  (west boundary)
    (-4337,  3639,    663,  3839),   # Wall top (north boundary)
    (-4360, -5670,    690, -5610),   # Wall bottom (south boundary)
    (-3127, -3081,  -2927,  1499),   # wall(5) internal N-S divider
    (-2997,   419,    523,   659),   # wall(6) kitchen shelf
    (-2927,   619,    473,  1319),   # wall(7) kitchen counter
    (-1772, -3231,    478, -3031),   # wall(4) corridor divider
    (-1517, -4648,  -1317, -3233),   # wall(3)
    (-2837, -4841,    663, -4641),   # wall(2) bottom horizontal
    (-2837, -5641,  -2637, -4641),   # wall(1) bottom corner
]

# Furniture bounding boxes (GPS-relative mm)
OBSTACLE_BOXES = [
    (-1957, -1701,   -957,   399),   # Bed   (2.1m×0.9m, rotated)
    ( -107, -3571,    293, -2671),   # Cabinet
    (-2437,  2929,  -1237,  3629),   # Desk
    (-1587,  2009,  -1087,  2509),   # Chair 1
    (-2637,  2009,  -2137,  2509),   # Chair 2
]


# NAVIGATION PARAMETERS

LOOKAHEAD_MM  = 200.0   # [v7] reduced from 350→200mm to suit dense 50mm wps
GOAL_R_MM     = 150.0   # arrival acceptance radius
KP_STEER      = 2.2     # [v7] slightly reduced for smoother turning

# [FIX 1] Updated lidar thresholds
CRUISE_V          = 0.18    # m/s  [v7] reduced from 0.20
TURN_V            = 0.05    # m/s  during sharp turns
LIDAR_STOP_M      = 0.18    # m    [v7] raised from 0.15 (was triggering too early)
LIDAR_SLOW_M      = 0.45    # m    [v7] raised from 0.35 (earlier warning)
LIDAR_TURN_M      = 0.22    # m    [v7] NEW: trigger TURN_AWAY recovery
LIDAR_CONE        = 20      # degrees either side of forward

# [FIX 1] Recovery motion parameters (NEW in v7)
RECOVER_W_RADS    = math.radians(38)   # rotation speed during TURN_AWAY
RECOVER_BACK_V    = -0.05              # reverse speed during REVERSE state
WAYPOINT_SPACING  = 50.0              # mm  [v7] dense path spacing (was 200mm)

N_RUNS   = 5
CSV_FILE = "astar_results.csv"


# HELPERS

def clamp(x, lo, hi):   return max(lo, min(hi, x))
def wrap180(a):          return (a + 180.0) % 360.0 - 180.0
def d2(ax, ay, bx, by): return math.hypot(bx - ax, by - ay)



# OCCUPANCY GRID 

class OccupancyGrid:
    """
    Binary occupancy grid built from Webots .wbt wall/furniture geometry.
    Row 0 = north (max Y).  Col 0 = west (min X).
    0 = free,  1 = obstacle (after inflation)
    """
    def __init__(self):
        self.res  = float(MAP_RES_MM)
        self.x1   = MAP_X1;  self.x2 = MAP_X2
        self.y1   = MAP_Y1;  self.y2 = MAP_Y2
        self.cols = int(math.ceil((MAP_X2 - MAP_X1) / MAP_RES_MM)) + 2
        self.rows = int(math.ceil((MAP_Y2 - MAP_Y1) / MAP_RES_MM)) + 2

        raw = [[False] * self.cols for _ in range(self.rows)]
        for bx1, by1, bx2, by2 in WALL_BOXES + OBSTACLE_BOXES:
            self._fill_rect(raw, bx1, by1, bx2, by2)

        rc      = int(math.ceil(INFLATE_MM / MAP_RES_MM))
        self.g  = self._inflate(raw, rc)

        occ = sum(c for row in self.g for c in row)
        print(f"[GRID] {self.cols}×{self.rows} cells  "
              f"res={MAP_RES_MM}mm  inflate={INFLATE_MM}mm  "
              f"occ={100*occ/(self.cols*self.rows):.1f}%")

    # ── Coordinate converters ────────────────────────────────────────────
    def w2g(self, x, y):
        """World mm → (col, row)."""
        c = int(round((x - self.x1) / self.res))
        r = int(round((self.y2 - y) / self.res))
        return c, r

    def g2w(self, c, r):
        """(col, row) → world mm."""
        return self.x1 + c * self.res, self.y2 - r * self.res

    def free(self, c, r):
        return 0 <= c < self.cols and 0 <= r < self.rows and not self.g[r][c]

    def snap(self, c, r, mr=60):
        """Snap (c,r) to nearest free cell."""
        if self.free(c, r):
            return c, r
        for d in range(1, mr + 1):
            for dr in range(-d, d + 1):
                for dc in range(-d, d + 1):
                    if max(abs(dr), abs(dc)) == d and self.free(c+dc, r+dr):
                        return c + dc, r + dr
        return None

    def _fill_rect(self, g, bx1, by1, bx2, by2):
        c1, r2 = self.w2g(bx1, by1)
        c2, r1 = self.w2g(bx2, by2)
        c1, c2 = min(c1, c2), max(c1, c2)
        r1, r2 = min(r1, r2), max(r1, r2)
        for r in range(max(0, r1), min(self.rows, r2 + 1)):
            for c in range(max(0, c1), min(self.cols, c2 + 1)):
                g[r][c] = True

    def _inflate(self, raw, rc):
        offs = [(dr, dc)
                for dr in range(-rc, rc + 1)
                for dc in range(-rc, rc + 1)
                if dr*dr + dc*dc <= rc*rc]
        out = [row[:] for row in raw]
        for r in range(self.rows):
            for c in range(self.cols):
                if raw[r][c]:
                    for dr, dc in offs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            out[nr][nc] = True
        return out



# A* PLANNER  

MOVES8 = [(-1, 0, 1.0), ( 1, 0, 1.0), (0, -1, 1.0), (0,  1, 1.0),
          (-1,-1, 1.414),(-1, 1, 1.414),( 1,-1, 1.414),( 1, 1, 1.414)]

def astar(grid, sc, sr, gc, gr):
    """8-connected A* with octile heuristic. Returns cell path or None."""
    def h(c, r):
        dx, dy = abs(gc - c), abs(gr - r)
        return max(dx, dy) + 0.414 * min(dx, dy)

    heap    = [(h(sc, sr), 0., sc, sr)]
    came    = {}
    g_score = {(sc, sr): 0.}
    visited = set()

    while heap:
        _, gc_cur, c, r = heapq.heappop(heap)
        if (c, r) in visited:
            continue
        if c == gc and r == gr:
            path = [(c, r)]
            while (c, r) in came:
                c, r = came[(c, r)]
                path.append((c, r))
            path.reverse()
            return path
        visited.add((c, r))
        for dc, dr, cost in MOVES8:
            nc, nr = c + dc, r + dr
            if not grid.free(nc, nr):
                continue
            ng = gc_cur + cost
            if ng < g_score.get((nc, nr), 1e18):
                g_score[(nc, nr)] = ng
                came[(nc, nr)]    = (c, r)
                heapq.heappush(heap, (ng + h(nc, nr), ng, nc, nr))
    return None

#  plan() — DENSE WAYPOINTS  
def plan(grid, sx, sy, gx, gy):

    sc, sr = grid.w2g(sx, sy)
    gc, gr = grid.w2g(gx, gy)
    sf = grid.snap(sc, sr)
    gf = grid.snap(gc, gr)
    if sf is None:
        raise RuntimeError("Start is inside obstacle")
    if gf is None:
        raise RuntimeError("Goal is inside obstacle — adjust GOAL_X/Y_MM")

    t0   = time.time()
    path = astar(grid, sf[0], sf[1], gf[0], gf[1])
    ms   = (time.time() - t0) * 1000.0

    if path is None:
        raise RuntimeError("A* found no path — try reducing INFLATE_MM")

    # Convert ALL grid cells to world coordinates (no RDP simplification)
    world = [grid.g2w(c, r) for c, r in path]

    # Dense interpolation: insert points every WAYPOINT_SPACING mm
    # so robot never skips more than 50mm between consecutive waypoints
    dense = [world[0]]
    for i in range(len(world) - 1):
        x1, y1 = world[i]
        x2, y2 = world[i + 1]
        seg = d2(x1, y1, x2, y2)
        if seg <= WAYPOINT_SPACING:
            dense.append((x2, y2))
        else:
            n = int(math.ceil(seg / WAYPOINT_SPACING))
            for k in range(1, n + 1):
                t = k / n
                dense.append((x1 + t * (x2 - x1),
                               y1 + t * (y2 - y1)))

    length = sum(d2(dense[i][0], dense[i][1],
                    dense[i+1][0], dense[i+1][1])
                 for i in range(len(dense) - 1))

    print(f"[A*] {len(path)} cells → {len(dense)} dense waypoints  "
          f"{length/1000:.2f}m  ({ms:.0f}ms)")
    return dense, length

# POSE ESTIMATOR 
class Pose:
    
    def __init__(self, gps, imu, el, er):
        self._gps = gps;  self._imu = imu
        self._el  = el;   self._er  = er
        self.x  = 0.0;  self.y  = 0.0;  self.th = 0.0
        self._ox = None; self._oy = None
        self._pl = None; self._pr = None
        self.mode = "init"

    def update(self):
        gv = self._gps.getValues()           if self._gps else [float("nan")] * 3
        iv = self._imu.getRollPitchYaw()     if self._imu else [float("nan")] * 3

        gps_ok = not any(math.isnan(v) or math.isinf(v) for v in gv)
        imu_ok = not any(math.isnan(v) or math.isinf(v) for v in iv)

        if gps_ok and imu_ok:
            gx, gy = gv[0], gv[1]           # Z-up world: x=East, y=North
            if self._ox is None:
                self._ox = gx;  self._oy = gy
                print(f"[POSE] GPS origin locked: ({gx:.4f},{gy:.4f}) m")
            self.x  = (gx - self._ox) * 1000.0
            self.y  = (gy - self._oy) * 1000.0
            self.th = math.degrees(iv[2]) % 360.0  # yaw around Z
            self.mode = "GPS+IMU"
            self._pl = self._el.getValue()
            self._pr = self._er.getValue()
            return

        # Wheel odometry fallback
        l = self._el.getValue();  r = self._er.getValue()
        if self._pl is None:
            self._pl = l;  self._pr = r;  self.mode = "odom";  return
        WR = WHEEL_R * 1000.0;  AX = AXLE * 1000.0
        dL = (l - self._pl) * WR;  dR = (r - self._pr) * WR
        self._pl = l;  self._pr = r
        ds   = (dR + dL) / 2.0
        dth  = (dR - dL) / AX
        th_m = math.radians(self.th) + dth / 2.0
        self.x  += ds * math.cos(th_m)
        self.y  += ds * math.sin(th_m)
        self.th  = (self.th + math.degrees(dth)) % 360.0
        self.mode = "odom"

# lidar_front() — 3-SECTOR LIDAR READING
def lidar_front(lidar):
    """
    [v7 CHANGE] Returns dict with 3 sectors instead of a single float.

      front  — minimum range in ±20° forward cone
      left   — minimum range in 20–60° left sector
      right  — minimum range in 20–60° right sector

    SickLms291: 180 points at 1°/point, index 90 = straight ahead.
    Having separate left/right readings lets PurePursuit choose
    which direction to rotate away from the obstacle.
    """
    result = {"front": 9.9, "left": 9.9, "right": 9.9}
    if lidar is None:
        return result
    try:
        rng = lidar.getRangeImage()
        if not rng:
            return result
        n   = len(rng)
        mid = n // 2   # index of straight-ahead beam

        def _sector_min(lo, hi):
            """Minimum valid range in rng[lo:hi+1] (metres)."""
            vals = [rng[i] for i in range(max(0, lo), min(n, hi + 1))
                    if 0.01 < rng[i] < 50.0
                    and not math.isnan(rng[i])
                    and not math.isinf(rng[i])]
            return min(vals) if vals else 9.9

        # Forward cone ±20°
        result["front"] = _sector_min(mid - 20, mid + 20)
        # Left sector 20–60° (indices BELOW mid in SickLms291 = left)
        result["left"]  = _sector_min(mid - 60, mid - 20)
        # Right sector 20–60° (indices ABOVE mid = right)
        result["right"] = _sector_min(mid + 20, mid + 60)

    except Exception:
        pass
    return result

#  PurePursuit — 4-STATE EXECUTION MACHINE
class PurePursuit:

    def __init__(self):
        self._idx         = 0
        self._state       = "MOVE"
        self._state_t     = 0.0
        self._recover_dir = 1.0    # +1 = turn left,  -1 = turn right

    def reset(self):
        self._idx         = 0
        self._state       = "MOVE"
        self._state_t     = 0.0
        self._recover_dir = 1.0

    def step(self, x, y, th, wps, lidar_data=9.9):
       
        # ── Accept both dict and plain float ─────────────────────────────
        if isinstance(lidar_data, dict):
            front = lidar_data.get("front", 9.9)
            left  = lidar_data.get("left",  9.9)
            right = lidar_data.get("right", 9.9)
        else:
            front = float(lidar_data)
            left  = 9.9
            right = 9.9

        # ── Goal check ────────────────────────────────────────────────────
        if not wps or self._idx >= len(wps):
            return 0., 0., "DONE", True

        gx, gy = wps[-1]
        dg = d2(x, y, gx, gy)
        if dg <= GOAL_R_MM:
            return 0., 0., f"GOAL err={dg:.0f}mm", True

        now = time.time()

        # STATE: REVERSE — back away from close obstacle
        if self._state == "REVERSE":
            elapsed = now - self._state_t
            if front > LIDAR_TURN_M or elapsed > 1.5:
                # Space opened or timed out → go back to turn
                self._state   = "TURN_AWAY"
                self._state_t = now
                return 0., 0., "REVERSE→TURN_AWAY", False
            return (RECOVER_BACK_V, 0.,
                    f"REVERSE f={front*1000:.0f}mm t={elapsed:.1f}s", False)

        
        # STATE: TURN_AWAY — rotate away from obstacle
    
        if self._state == "TURN_AWAY":
            elapsed = now - self._state_t

            # Check for exit
            if front > LIDAR_SLOW_M or elapsed > 3.0:
                self._state   = "MOVE"
                self._state_t = now
                return 0., 0., f"TURN_AWAY→MOVE f={front*1000:.0f}mm", False

            # Still very close → switch to reverse
            if front < LIDAR_STOP_M:
                self._state   = "REVERSE"
                self._state_t = now
                return 0., 0., f"TURN_AWAY→REVERSE f={front*1000:.0f}mm", False

            # Choose turn direction: away from closer side
            if abs(left - right) > 0.05:
                turn_dir = -1.0 if left < right else 1.0
            else:
                # Head-on: use stored preference
                turn_dir = self._recover_dir

            w = turn_dir * RECOVER_W_RADS
            # Tiny creep forward if enough room (helps around corners)
            v = 0.03 if front > LIDAR_TURN_M * 1.4 else 0.0
            side = "L" if turn_dir > 0 else "R"
            return (v, w,
                    f"TURN_AWAY {side} f={front*1000:.0f}mm "
                    f"l={left*1000:.0f} r={right*1000:.0f}", False)

        
        # Obstacle detection → switch to recovery states
       
        if front < LIDAR_STOP_M:
            # Very close: reverse immediately
            self._state   = "REVERSE"
            self._state_t = now
            return 0., 0., f"→REVERSE f={front*1000:.0f}mm", False

        if front < LIDAR_TURN_M:
            # Close enough to start turning away
            # Choose direction based on which side is more open
            if abs(left - right) > 0.05:
                self._recover_dir = -1.0 if left < right else 1.0
            else:
                # Default: turn toward more open side using path direction
                desired  = math.degrees(math.atan2(
                    wps[min(self._idx, len(wps)-1)][1] - y,
                    wps[min(self._idx, len(wps)-1)][0] - x
                )) % 360.0
                err_sign = wrap180(desired - th)
                self._recover_dir = 1.0 if err_sign > 0 else -1.0
            self._state   = "TURN_AWAY"
            self._state_t = now
            return 0., 0., f"→TURN_AWAY f={front*1000:.0f}mm", False

        
        # STATES: MOVE / SLOW — normal pure-pursuit path following
        
        self._state = "SLOW" if front < LIDAR_SLOW_M else "MOVE"

        # Advance past already-passed dense waypoints
        # With 50mm spacing use a tighter threshold than before
        while self._idx < len(wps) - 1:
            wx, wy = wps[self._idx]
            if d2(x, y, wx, wy) < LOOKAHEAD_MM * 0.40:
                self._idx += 1
            else:
                break

        # Find lookahead point: first waypoint ≥ LOOKAHEAD_MM ahead
        lx, ly = wps[min(self._idx, len(wps) - 1)]
        for j in range(self._idx, len(wps)):
            wx, wy = wps[j]
            if d2(x, y, wx, wy) >= LOOKAHEAD_MM:
                lx, ly = wx, wy
                break

        # Heading error → angular velocity (P-control)
        desired = math.degrees(math.atan2(ly - y, lx - x)) % 360.0
        err_d   = wrap180(desired - th)
        err_r   = math.radians(err_d)
        w       = clamp(KP_STEER * err_r, -MAX_W, MAX_W)

        # Forward velocity — reduce for sharp turns
        alpha  = max(0.0, 1.0 - abs(err_r) / (math.pi / 2))
        v_turn = TURN_V + (CRUISE_V - TURN_V) * alpha

        # Additional slow-down near obstacles (smooth ramp)
        if front < LIDAR_SLOW_M:
            ratio = ((front - LIDAR_TURN_M) /
                     (LIDAR_SLOW_M - LIDAR_TURN_M))
            v_obs = max(0.03, v_turn * clamp(ratio, 0.1, 1.0))
        else:
            v_obs = v_turn

        info = (f"WP{self._idx+1}/{len(wps)} "
                f"err={err_d:+.0f}° d={dg:.0f}mm "
                f"f={front*1000:.0f}mm [{self._state}]")
        return v_obs, w, info, False



# DRIVE 

def drive(lm, rm, v, w):
    """Convert (v m/s, w rad/s) → individual wheel velocities."""
    h  = AXLE / 2.0
    vl = (v - w * h) / WHEEL_R
    vr = (v + w * h) / WHEEL_R
    lm.setVelocity(clamp(vl, -MAX_WHEEL, MAX_WHEEL))
    rm.setVelocity(clamp(vr, -MAX_WHEEL, MAX_WHEEL))



# CSV LOGGER  

def init_csv():
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "run", "event", "mode",
            "x_mm", "y_mm", "th_deg",
            "goal_x_mm", "goal_y_mm", "error_mm",
            "time_s", "path_mm", "n_waypoints", "plan_ms"
        ])

def log_row(run, ev, pose, t, plen, nwp, pms):
    err = d2(pose.x, pose.y, GOAL_X_MM, GOAL_Y_MM)
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            run, ev, pose.mode,
            round(pose.x, 1), round(pose.y, 1), round(pose.th, 1),
            GOAL_X_MM, GOAL_Y_MM, round(err, 1),
            round(t, 2), round(plen, 1), nwp, round(pms, 1)
        ])

# MAIN

def main():
    robot = Robot()

    # ── Enable ALL sensors BEFORE first robot.step() (prevents GPS nan) ──
    lm = robot.getDevice(MOTOR_L);  rm = robot.getDevice(MOTOR_R)
    if not lm or not rm:
        raise RuntimeError(f"Motors not found: '{MOTOR_L}' / '{MOTOR_R}'")
    lm.setPosition(float("inf"));   rm.setPosition(float("inf"))
    lm.setVelocity(0.0);            rm.setVelocity(0.0)
    print("[INIT] ✓ Motors: 'left wheel' / 'right wheel'")

    el = robot.getDevice(ENC_L);    er = robot.getDevice(ENC_R)
    el.enable(TIME_STEP);           er.enable(TIME_STEP)
    print("[INIT] ✓ Encoders")

    gps = robot.getDevice(GPS_N)
    if gps:  gps.enable(TIME_STEP);  print("[INIT] ✓ GPS")
    else:    print("[INIT] ✗ GPS missing — using odometry only")

    imu = robot.getDevice(IMU_N)
    if imu:  imu.enable(TIME_STEP);  print("[INIT] ✓ InertialUnit")
    else:    print("[INIT] ✗ IMU missing")

    lidar = robot.getDevice(LIDAR_N)
    if lidar:
        lidar.enable(TIME_STEP);  lidar.enablePointCloud()
        print(f"[INIT] ✓ Lidar '{LIDAR_N}'")
    else:
        print("[INIT] ✗ Lidar missing — obstacle avoidance disabled")

    sonars = []
    for name in SONARS:
        s = robot.getDevice(name)
        if s:  s.enable(TIME_STEP);  sonars.append(s)
    print(f"[INIT] ✓ Sonars: {len(sonars)}")

    # ── 25 warm-up steps so GPS/IMU values settle ─────────────────────────
    print("[INIT] Warming up 25 steps...")
    for _ in range(25):
        robot.step(TIME_STEP)

    # ── Read start pose from GPS ───────────────────────────────────────────
    pose = Pose(gps, imu, el, er)
    pose.update()
    print(f"\n[ROBOT] Start:  ({pose.x:.0f}, {pose.y:.0f}) mm  "
          f"θ={pose.th:.1f}°  [{pose.mode}]")
    print(f"[ROBOT] Goal:   ({GOAL_X_MM:.0f}, {GOAL_Y_MM:.0f}) mm")
    print(f"[ROBOT] S-line: "
          f"{d2(pose.x, pose.y, GOAL_X_MM, GOAL_Y_MM)/1000:.2f} m\n")

    if pose.mode == "odom":
        print("[WARN] GPS returned nan — running on wheel odometry.")
        print("[WARN] Accuracy will degrade over long distances.\n")

    # ── Build occupancy grid from .wbt wall geometry ───────────────────────
    grid = OccupancyGrid()

    # Verify start and goal snapped to free cells
    sf = grid.snap(*grid.w2g(pose.x, pose.y))
    gf = grid.snap(*grid.w2g(GOAL_X_MM, GOAL_Y_MM))
    if sf is None:
        print("[ERROR] Start is inside solid obstacle"); return
    if gf is None:
        print("[ERROR] Goal is inside solid obstacle — adjust GOAL_X/Y_MM"); return

    sx_w, sy_w = grid.g2w(*sf)
    gx_w, gy_w = grid.g2w(*gf)
    print(f"[GRID] Start snapped to ({sx_w:.0f}, {sy_w:.0f}) mm")
    print(f"[GRID] Goal  snapped to ({gx_w:.0f}, {gy_w:.0f}) mm")

    # ── Plan initial A* route ──────────────────────────────────────────────
    t_plan = time.time()
    try:
        wps, plen = plan(grid, pose.x, pose.y, GOAL_X_MM, GOAL_Y_MM)
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("[HINT] Reduce INFLATE_MM or check GOAL_X_MM / GOAL_Y_MM")
        return
    plan_ms = (time.time() - t_plan) * 1000.0

    # Print first/last few waypoints (not all 291)
    print(f"\n[ROUTE] {len(wps)} dense waypoints  ({plen/1000:.2f} m):")
    shown = min(8, len(wps))
    for i in range(shown):
        print(f"  WP{i+1:03d}  ({wps[i][0]:7.0f}, {wps[i][1]:7.0f}) mm")
    if len(wps) > shown * 2:
        print(f"  ...  ({len(wps) - shown * 2} more) ...")
    for i in range(max(shown, len(wps) - shown), len(wps)):
        print(f"  WP{i+1:03d}  ({wps[i][0]:7.0f}, {wps[i][1]:7.0f}) mm")

    # ── Experiment loop ────────────────────────────────────────────────────
    init_csv()
    pp = PurePursuit()
    PRINT_N = max(1, int(2000 / TIME_STEP))   # print every ~2 s

    for run in range(1, N_RUNS + 1):
        print(f"\n{'━'*60}")
        print(f"  RUN {run}/{N_RUNS}  |  goal=({GOAL_X_MM:.0f},{GOAL_Y_MM:.0f})mm")
        print(f"{'━'*60}")

        # Re-plan from current GPS position each run (handles any drift)
        pose.update()
        try:
            tp = time.time()
            wps, plen = plan(grid, pose.x, pose.y, GOAL_X_MM, GOAL_Y_MM)
            plan_ms = (time.time() - tp) * 1000.0
        except RuntimeError as e:
            print(f"[WARN] Re-plan failed: {e} — using previous plan")

        pp.reset()
        t0   = time.time()
        tick = 0
        done = False
        log_row(run, "START", pose, 0., plen, len(wps), plan_ms)

        while robot.step(TIME_STEP) != -1:
            pose.update()

            # [FIX 3] lidar_front() now returns dict {front,left,right}
            fm = lidar_front(lidar)

            # [FIX 4] PurePursuit.step() accepts the dict
            v, w, info, done = pp.step(pose.x, pose.y, pose.th, wps, fm)
            drive(lm, rm, v, w)

            tick += 1
            if tick % PRINT_N == 0:
                elapsed = time.time() - t0
                dg = d2(pose.x, pose.y, GOAL_X_MM, GOAL_Y_MM)
                # [FIX 3] Updated print: show front/left/right separately
                print(f"  t={elapsed:5.1f}s  "
                      f"({pose.x:6.0f},{pose.y:6.0f})mm  "
                      f"θ={pose.th:5.1f}°  "
                      f"d={dg:5.0f}mm  "
                      f"f={fm['front']*1000:.0f}mm "
                      f"L={fm['left']*1000:.0f}mm "
                      f"R={fm['right']*1000:.0f}mm  "
                      f"[{pose.mode}]  {info}")

            if done:
                drive(lm, rm, 0., 0.)
                pose.update()
                elapsed = time.time() - t0
                err     = d2(pose.x, pose.y, GOAL_X_MM, GOAL_Y_MM)
                print(f"\n  ✓ RUN {run} COMPLETE")
                print(f"    Final error  : {err:.0f} mm")
                print(f"    Time         : {elapsed:.1f} s")
                print(f"    Path length  : {plen/1000:.2f} m")
                print(f"    Waypoints    : {len(wps)}")
                print(f"    Pose mode    : {pose.mode}")
                log_row(run, "ARRIVED", pose, elapsed, plen, len(wps), plan_ms)
                break

        if not done:
            drive(lm, rm, 0., 0.)
            pose.update()
            elapsed = time.time() - t0
            print(f"  ✗ RUN {run} TIMEOUT  t={elapsed:.0f}s")
            log_row(run, "TIMEOUT", pose, elapsed, plen, len(wps), plan_ms)

        # 3-second pause between runs
        drive(lm, rm, 0., 0.)
        for _ in range(int(3000 / TIME_STEP)):
            if robot.step(TIME_STEP) == -1:
                return

    print(f"\n[DONE] {N_RUNS} runs complete → {CSV_FILE}")


if __name__ == "__main__":
    main()