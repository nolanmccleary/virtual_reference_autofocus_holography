"""Walk scene: handheld motion-based synthetic aperture.

A "laptop" (object center) carries a scan antenna and a reference antenna at
fixed rigid offsets. It performs a seeded 6-DOF random walk (translation +
pitch/roll/yaw) through the WALK MESH (a box behind the recording plane) while
illuminated by the router. Perfect IMU gives the exact antenna pose each step.

At each step the scan/reference antennas sample the field; the homodyne ratio
cancels the router's clock skew; a reference correction (using the known ref
pose) pins the reference to a virtual fixed position so all samples are mutually
comparable. The scene is reconstructed by coherent BACKPROJECTION over the true
3D sample positions.
"""

from __future__ import annotations

import numpy as np

from ..scene import CrossScene
from ..propagation import angular_spectrum_propagate as _asp
from ..sources import point_source_field, wavelength_of, box_image_sources
from .base import Built, param, defaults_from_schema

NAME = "walk"
LABEL = "Walk (handheld synthetic aperture)"

PARAM_SCHEMA = [
    param("carrier_ghz", "Carrier", "Wave", default=5.0, lo=2.0, hi=6.0, step=0.1, fmt="ghz"),
    param("bandwidth_mhz", "Bandwidth", "Wave", default=70, lo=0, hi=200, step=5, fmt="mhz"),
    param("n_freq", "Freq. samples", "Wave", default=8, lo=1, hi=24, step=1, fmt="int"),
    param("nx", "Recon resolution", "Grid", default=80, lo=48, hi=140, step=4, fmt="px"),
    param("z_emitter_cm", "Router Z", "Router", default=230, lo=120, hi=350, step=5, fmt="cm"),
    param("emitter_x_cm", "Router X", "Router", default=150, lo=0, hi=300, step=5, fmt="cm"),
    param("emitter_y_cm", "Router Y", "Router", default=100, lo=0, hi=200, step=5, fmt="cm"),
    param("z_object_cm", "Cross Z", "Cross", default=140, lo=30, hi=210, step=5, fmt="cm"),
    param("cross_arm_cm", "Cross arm", "Cross", default=100, lo=20, hi=180, step=5, fmt="cm"),
    param("cross_thickness_cm", "Cross thickness", "Cross", default=5, lo=0, hi=60, step=1, fmt="cm"),
    param("walk_depth_cm", "Walk-mesh depth", "Walk mesh", default=100, lo=20, hi=250, step=10, fmt="cm"),
    param("center_x_cm", "Center start X", "Walk mesh", default=150, lo=0, hi=300, step=5, fmt="cm"),
    param("center_y_cm", "Center start Y", "Walk mesh", default=100, lo=0, hi=200, step=5, fmt="cm"),
    param("center_z_cm", "Raster depth Z", "Walk mesh", default=-10, lo=-240, hi=0, step=5, fmt="cm"),
    param("scan_off_x_cm", "Scan offset X", "Scan antenna", default=0, lo=-60, hi=60, step=2, fmt="cm"),
    param("scan_off_y_cm", "Scan offset Y", "Scan antenna", default=0, lo=-60, hi=60, step=2, fmt="cm"),
    param("scan_off_z_cm", "Scan offset Z", "Scan antenna", default=0, lo=-60, hi=60, step=2, fmt="cm"),
    param("ref_off_x_cm", "Ref offset X", "Reference antenna", default=-30, lo=-60, hi=60, step=2, fmt="cm"),
    param("ref_off_y_cm", "Ref offset Y", "Reference antenna", default=0, lo=-60, hi=60, step=2, fmt="cm"),
    param("ref_off_z_cm", "Ref offset Z", "Reference antenna", default=10, lo=-60, hi=60, step=2, fmt="cm"),
    # --- walk pattern + its pattern-specific control ---
    param("pattern", "Walk pattern", "Walk", kind="select", options=["raster", "spiral"], default="raster"),
    # raster: # of zigzag rows (shows for pattern=raster) -- images the cross
    param("raster_rows", "Zigzag rows", "Walk", default=16, lo=2, hi=32, step=1, fmt="int", show_if={"pattern": "raster"}),
    # spiral: # of loops = sweep distance (shows for pattern=spiral)
    param("sweep_loops", "Spiral loops", "Walk", default=4, lo=1, hi=16, step=1, fmt="int", show_if={"pattern": "spiral"}),
    # walk steps = # of random-walk knots (path randomness / hand error granularity)
    param("n_steps", "Walk steps", "Walk", default=40, lo=2, hi=300, step=1, fmt="int"),
    # samples = how many measurements, uniform along the trajectory (system speed)
    param("num_samples", "Samples", "Walk", default=4000, lo=100, hi=8000, step=100, fmt="int"),
    param("jitter_cm", "Hand jitter", "Walk", default=10, lo=0, hi=40, step=1, fmt="cm"),
    param("tilt_deg", "Tilt amplitude", "Walk", default=10, lo=0, hi=40, step=1, fmt="int"),
    param("seed", "Random seed", "Walk", default=67, lo=0, hi=999, step=1, fmt="int"),
    param("ref_correction", "Reference correction", "Walk", kind="toggle", default=True),
    param("multipath", "Multipath", "Multipath", kind="toggle", default=False),
    param("wall_distance_m", "Wall distance", "Multipath", default=2.5, lo=0.5, hi=6.0, step=0.1, fmt="m"),
    param("reflectivity", "Reflectivity", "Multipath", default=0.5, lo=0.0, hi=1.0, step=0.05, fmt="num"),
]

PLANE_X, PLANE_Y = 3.0, 2.0   # recording / reconstruction footprint (m)
N_WZ = 18                      # depth planes used to sample the field in the walk mesh


def defaults() -> dict:
    return defaults_from_schema(PARAM_SCHEMA)


def _g(p, k):
    return p.get(k, next(s["default"] for s in PARAM_SCHEMA if s["key"] == k))


def _rot(angles):
    """Rotation matrix from (roll, pitch, yaw) radians: Rz·Ry·Rx."""
    rx, ry, rz = angles
    cx, sx = np.cos(rx), np.sin(rx); cy, sy = np.cos(ry), np.sin(ry); cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _reflect(c, bounds):
    """Reflect a point back inside axis-aligned bounds [(lo,hi)×3]."""
    out = c.copy()
    for i, (lo, hi) in enumerate(bounds):
        if hi <= lo:
            out[i] = lo; continue
        span = hi - lo
        t = (out[i] - lo) % (2 * span)
        out[i] = lo + (t if t <= span else 2 * span - t)
    return out


def _sampler(planes, zs, X, Y, Lx, Ly, nx, ny, dx, dy):
    """Return a function sampling a precomputed plane-stack at a 3D point
    (nearest plane in z, bilinear in x,y)."""
    def sample(pt):
        px = min(max(pt[0], 0.0), Lx - 1e-6); py = min(max(pt[1], 0.0), Ly - 1e-6)
        iz = int(np.argmin(np.abs(zs - pt[2])))
        gx = px / dx - 0.5; gy = py / dy - 0.5
        x0 = int(np.clip(np.floor(gx), 0, nx - 2)); y0 = int(np.clip(np.floor(gy), 0, ny - 2))
        fx = np.clip(gx - x0, 0, 1); fy = np.clip(gy - y0, 0, 1)
        f = planes[iz]
        return ((f[y0, x0] * (1 - fx) + f[y0, x0 + 1] * fx) * (1 - fy)
                + (f[y0 + 1, x0] * (1 - fx) + f[y0 + 1, x0 + 1] * fx) * fy)
    return sample


def build(p: dict) -> Built:
    Lx, Ly = PLANE_X, PLANE_Y
    nx = int(_g(p, "nx")); dx = Lx / nx; ny = max(1, int(round(Ly / dx))); dy = Ly / ny
    z_emit = float(_g(p, "z_emitter_cm")) / 100.0
    z_obj = float(_g(p, "z_object_cm")) / 100.0
    em_xy = (float(_g(p, "emitter_x_cm")) / 100.0, float(_g(p, "emitter_y_cm")) / 100.0)
    emitter_xyz = (em_xy[0], em_xy[1], z_emit)

    carrier = float(_g(p, "carrier_ghz")) * 1e9
    bw = float(_g(p, "bandwidth_mhz")) * 1e6
    nf = int(_g(p, "n_freq"))
    freqs = (np.array([carrier]) if nf <= 1 else np.linspace(carrier - bw / 2, carrier + bw / 2, nf))
    carrier_idx = int(np.argmin(np.abs(freqs - carrier)))

    # cross object mask (reuse CrossScene geometry helpers)
    cs = CrossScene(Lx=Lx, Ly=Ly, dx=dx, z_object=z_obj, z_emitter=z_emit,
                    carrier_hz=carrier, cross_arm_length=float(_g(p, "cross_arm_cm")) / 100.0,
                    cross_thickness=float(_g(p, "cross_thickness_cm")) / 100.0,
                    emitter_xy=em_xy, cross_center_xy=(Lx / 2, Ly / 2))
    mask = cs.cross_mask()
    x = (np.arange(nx) + 0.5) * dx; y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)

    # ---- DECOUPLED motion model (a person scanning a laptop) ----
    #  spiral loops  -> how far the hand sweeps (trajectory distance/extent)
    #  walk steps    -> # of random-walk knots inside the spiral (muscle state
    #                   changes; path randomness granularity)
    #  samples       -> # of measurements taken UNIFORMLY along the trajectory,
    #                   independent of the above (set by system/hardware speed)
    seed = int(_g(p, "seed")); rng = np.random.default_rng(seed)
    pattern = str(_g(p, "pattern"))
    n_knots = max(2, int(_g(p, "n_steps")))
    N = max(2, int(_g(p, "num_samples")))
    jitter = float(_g(p, "jitter_cm")) / 100.0
    tilt = np.deg2rad(float(_g(p, "tilt_deg")))
    walk_depth = float(_g(p, "walk_depth_cm")) / 100.0
    c0 = np.array([float(_g(p, "center_x_cm")), float(_g(p, "center_y_cm")), float(_g(p, "center_z_cm"))]) / 100.0
    scan_off = np.array([float(_g(p, "scan_off_x_cm")), float(_g(p, "scan_off_y_cm")), float(_g(p, "scan_off_z_cm"))]) / 100.0
    ref_off = np.array([float(_g(p, "ref_off_x_cm")), float(_g(p, "ref_off_y_cm")), float(_g(p, "ref_off_z_cm"))]) / 100.0
    bounds = [(0.0, Lx), (0.0, Ly), (-walk_depth, 0.0)]

    # the continuous trajectory, evaluated at the N sample positions
    fs = np.linspace(0.0, 1.0, N)
    if pattern == "raster":
        # boustrophedon zigzag across the plane at ~constant depth (a person
        # "painting" the plane in rows) -> uniform planar aperture -> images the cross
        rows = max(2, int(_g(p, "raster_rows")))
        tri = 2 * np.abs((fs * rows) % 1.0 - 0.5)    # 0→1 triangle, `rows` periods
        base = np.column_stack([
            0.08 * Lx + 0.84 * Lx * tri,
            0.08 * Ly + 0.84 * Ly * fs,
            c0[2] + np.zeros(N),                      # held at ~constant depth
        ])
    else:
        # spiral sweep (emitter localization; does NOT image the cross cleanly)
        loops = max(1, int(_g(p, "sweep_loops")))
        theta = 2 * np.pi * loops * fs
        env = np.sin(np.pi * fs); rad = 0.12 + 0.88 * env
        ax = 0.42 * Lx; ay = 0.42 * Ly; az = 0.42 * walk_depth
        base = np.column_stack([
            c0[0] + ax * rad * np.cos(theta),
            c0[1] + ay * rad * np.sin(theta),
            c0[2] + az * np.sin(2 * np.pi * 1.5 * fs),
        ])
    theta = 2 * np.pi * max(1, int(_g(p, "sweep_loops"))) * fs   # used for tilt wobble

    # random-walk perturbation defined at n_knots knots, interpolated to N samples
    fk = np.linspace(0.0, 1.0, n_knots)
    def _ou_knots(sigma):
        out = np.zeros((n_knots, 3)); d = np.zeros(3)
        for i in range(n_knots):
            d = 0.85 * d + rng.normal(0, sigma, 3); out[i] = d
        return out
    def _interp_to_samples(knots):
        return np.column_stack([np.interp(fs, fk, knots[:, a]) for a in range(3)])

    jit_s = _interp_to_samples(_ou_knots(jitter * 0.45))
    centers = np.array([_reflect(base[t] + jit_s[t], bounds) for t in range(N)])

    tilt_base = tilt * np.column_stack([
        np.sin(1.3 * theta + 0.5), np.sin(0.9 * theta + 1.1), np.sin(1.7 * theta)])
    tilt_s = tilt_base + _interp_to_samples(_ou_knots(tilt * 0.12))

    skew = np.exp(1j * rng.uniform(0, 2 * np.pi, N))   # clock skew per measurement
    obj_path = centers
    scan_path = np.array([centers[t] + _rot(tilt_s[t]) @ scan_off for t in range(N)])
    ref_path = np.array([centers[t] + _rot(tilt_s[t]) @ ref_off for t in range(N)])
    n_steps = N   # downstream sampling loop iterates over the N measurements

    # ---- precompute the total field on N_WZ planes spanning the antenna z-range ----
    all_z = np.concatenate([scan_path[:, 2], ref_path[:, 2]])
    zlo, zhi = all_z.min() - 0.05, all_z.max() + 0.05
    zs = np.linspace(zlo, zhi, N_WZ)

    ref_correct = bool(_g(p, "ref_correction"))
    ex_, ey_, ez_ = emitter_xyz

    # multipath via the SHARED box-image-source helper (walls `wall_distance_m`
    # beyond the scene, like the cross scene). Image sources are analytic spherical
    # waves added directly at the sampled 3D point.
    multipath = bool(_g(p, "multipath"))
    images = []
    if multipath:
        m = float(_g(p, "wall_distance_m"))
        box = (-m, Lx + m, -m, Ly + m, -m, z_emit + m)
        images = box_image_sources(emitter_xyz, box, float(_g(p, "reflectivity")))

    def field_fn(freq):
        """Total field at an arbitrary 3D point: analytic direct emitter wave
        (exact) + cross scatter (propagated planes) + multipath image sources."""
        wl = wavelength_of(freq); kw = 2.0 * np.pi / wl
        obj_nomask = point_source_field(X, Y, z_obj, emitter_xyz, wl)
        scatter = [_asp(obj_nomask * mask - obj_nomask, dz=(z_obj - zp),
                        wavelength=wl, dx=dx, dy=dy) for zp in zs]
        samp_scatter = _sampler(scatter, zs, X, Y, Lx, Ly, nx, ny, dx, dy)

        def F(a):
            R = np.sqrt((a[0] - ex_) ** 2 + (a[1] - ey_) ** 2 + (a[2] - ez_) ** 2)
            val = np.exp(1j * kw * R) / max(R, 1e-6) + samp_scatter(a)
            for (ix, iy, iz), amp in images:
                Ri = np.sqrt((a[0] - ix) ** 2 + (a[1] - iy) ** 2 + (a[2] - iz) ** 2)
                val = val + amp * np.exp(1j * kw * Ri) / max(Ri, 1e-6)
            return val
        return F

    sample_val = []
    for f in freqs:
        F = field_fn(f)
        Fref0 = F(ref_path[0])
        vals = np.zeros(n_steps, dtype=complex)
        for t in range(n_steps):
            Fs = F(scan_path[t]); Fr = F(ref_path[t])
            if abs(Fr) < 1e-9:
                Fr = 1e-9
            ratio = (skew[t] * Fs) / (skew[t] * Fr)          # homodyne: cancels clock skew
            Fref_used = Fr if ref_correct else Fref0          # correction pins ref to true pose
            vals[t] = ratio * Fref_used
        sample_val.append(vals)

    z_values = np.linspace(0.3, z_emit + 0.3, 41)
    geometry = {
        "kind": "walk",
        "Lx": Lx, "Ly": Ly, "zEmitter": z_emit, "zObject": z_obj,
        "emitterXY": list(em_xy),
        "crossCenter": [Lx / 2, Ly / 2], "armL": cs.cross_arm_length / 2.0,
        "armW": cs.cross_arm_width / 2.0, "crossThickness": cs.cross_thickness,
        "occluders": [{"z": z_obj, "kind": "cross", "cx": Lx / 2, "cy": Ly / 2,
                       "L": cs.cross_arm_length / 2.0, "w": cs.cross_arm_width / 2.0}],
        "walkDepth": walk_depth, "multipath": multipath,
        "room": ({"x0": -float(_g(p, "wall_distance_m")), "x1": Lx + float(_g(p, "wall_distance_m")),
                  "y0": -float(_g(p, "wall_distance_m")), "y1": Ly + float(_g(p, "wall_distance_m")),
                  "zFront": -float(_g(p, "wall_distance_m")), "zBack": z_emit + float(_g(p, "wall_distance_m"))}
                 if multipath else None),
        "objPath": obj_path.tolist(), "scanPath": scan_path.tolist(), "refPath": ref_path.tolist(),
        "refCorrection": ref_correct,
    }
    return Built(
        Lx=Lx, Ly=Ly, dx=dx, dy=dy, nx=nx, ny=ny,
        z_values=z_values, z_emitter=z_emit,
        emitter_xy=em_xy, reference_xy=tuple(ref_path[0, :2]),
        freqs=freqs, carrier_idx=carrier_idx, carrier_wavelength=wavelength_of(carrier),
        backend="backprojection",
        sample_pos=scan_path, sample_val=sample_val,
        recon_x=x, recon_y=y, geometry=geometry,
        scene_extra={"iEmitter": int(np.argmin(np.abs(z_values - z_emit)))},
    )
