# Algorithms reference (verification spec)

The authoritative mathematical definition of every numerical algorithm in the
engine. Use this to verify any port (e.g. numpy → PyTorch/MPS) is **bit-for-bit
mathematically aligned**: the port must implement these exact formulas.

Conventions used throughout:
- Time convention `exp(-iωt)`; a **diverging** spherical wave is `exp(+ikR)/R`.
- `k = 2π/λ` is the angular wavenumber; `λ = c/f`, `c = 299792458 m/s`.
- Fields are complex; intensity is `|field|²`.
- Grids are indexed `[y, x]`; `dx, dy` are sample spacings (m).

---

## 1. Angular-spectrum propagation  `propagation.angular_spectrum_propagate(U, dz, λ, dx, dy)`

Propagates a complex field `U(x,y)` on a plane by a distance `dz` (dz>0 =
forward/away from source; dz<0 = backward). Zero-padded to 2× (then cropped) to
suppress circular wrap-around.

1. Spatial frequencies: `fx = fftfreq(Nx, dx)`, `fy = fftfreq(Ny, dy)`; grid `(FX, FY)`.
2. `κ = 1/λ`  (plane-wave spatial-frequency magnitude).
3. `arg = κ² − FX² − FY²`.
4. Transfer function (evanescent components **decay**, never blow up):
   ```
   H = exp( i · 2π · dz · sqrt(arg) )          where arg ≥ 0   (propagating)
   H = exp( −2π · |dz| · sqrt(−arg) )          where arg < 0   (evanescent)
   ```
5. Output: `U' = IFFT2( FFT2(U_padded) · H )`, cropped back to the original size.

---

## 2. Point-source field  `sources.point_source_field(X, Y, z_plane, E, λ)`

Exact scalar free-space Green's function sampled on a plane at `z_plane` from an
emitter at `E = (ex, ey, ez)`:
```
R(x,y)   = sqrt( (x−ex)² + (y−ey)² + (z_plane−ez)² )
F(x,y)   = exp( i · (2π/λ) · R ) / R
```

---

## 3. Forward hologram — cross scene  `forward.record_hologram(scene, f)`

Complex field recorded on the scan plane (z=0) for one frequency.

Finite-thickness cross = short stack of thin absorbing masks over
`[z_object − T/2, z_object + T/2]`:
- `planes = linspace(z_object + T/2, z_object − T/2, n)`, `n = clip(round(T/0.05)+1, 2, 6)`; if `T ≤ 0`, `planes = [z_object]`.
- Launch: `u = point_source(X, Y, planes[0], E, λ)`.
- For each plane `zp` (far→near): `u ← u · mask`; then `u ← asp(u, dz = zp − next_z, λ, dx, dy)` where `next_z` is the next plane or `0` for the last.

`mask(x,y) = 0` inside the opaque cross footprint, `1` elsewhere. Cross footprint
= union of two bars (half-length `L = arm/2`, half-width `w = width/2`) centred at
`(cx, cy)`:
```
horiz: |x−cx| ≤ L AND |y−cy| ≤ w
vert : |y−cy| ≤ L AND |x−cx| ≤ w
```

**Multipath** (if enabled): add first-order wall image sources, each a point
source from a mirrored emitter position with amplitude `r` (reflectivity):
```
u ← u + Σ_images  r · point_source(X, Y, 0, image_pos, λ)
```
Image positions (6 walls, margins `m` beyond the scene; back/front at `z_emitter+room_back`, `−room_front`):
`(2·(−m)−ex, ey, ez)`, `(2·(Lx+m)−ex, ey, ez)`, `(ex, 2·(−m)−ey, ez)`, `(ex, 2·(Ly+m)−ey, ez)`, `(ex, ey, 2·zBack−ez)`, `(ex, ey, 2·zFront−ez)`.

## 3b. Forward field — room scene  `scenes/room._record`
Same multi-slice idea with rectangular shelf masks at their z-planes (far→near),
emitter on the floor at `z=height`, recording plane the ceiling at `z=0`; plus 5
wall image sources (4 walls + ceiling) when multipath on.

---

## 4. Homodyne normalization (Eq. 1 of the paper)

Cancels the emitter's unknown signal `U_em(f)` (clock skew) by dividing the
scanned signal by a reference antenna's signal:
```
I·e^{iΦ}|_{x,y} = F{ I(t)|_{x,y} } / F{ I(t)|_{ref} }
```
In the simulation this is: `holo = U / U_ref`, where `U_ref` is the full field
sampled at the reference-antenna pixel. (A global constant `U(ref)⁻¹` remains,
which is irrelevant to the image.)

---

## 5. Synthetic-aperture scan grid (block-fill)  `engine.set_params`

Two grids over the plane:
- **slice resolution** `nx × ny` (fine compute/display grid, `dx = Lx/nx`).
- **scan grid** `nx_pts × ny_pts` measurement points at separation `sep`
  (`nx_pts = round(Lx/sep)`, clamped so `sep ≥ dx`).

Each fine pixel is "owned" by its nearest scan point. The **measured** field is
the full field sampled (block-filled) at the owning scan point:
```
U_measured[y,x] = U_full[ srow[y], scol[x] ]      (nearest scan point)
owner_index[y,x] = owner_row[y]·nx_pts + owner_col[x]
```
Partial aperture after `k` acquired scan points (raster order):
```
aperture_mask = (owner_index < k)   ∈ {0,1}
```

---

## 6. Reconstruction A — angular-spectrum backpropagation  `engine._stack / _recon_one`

`holo_f = (U_f · aperture_mask) / ref_f`.  Backpropagate to depth `z`:
`bp(holo, z) = asp(holo, dz = −z, λ, dx, dy)`.

- **single** (carrier `fc`): `I(x,y,z) = |bp(holo_fc, z)|²`.
- **white-light**: `I(x,y,z) = Σ_f |bp(holo_f, z)|²`.
- **dark-field**: `I = clip( Σ_f |illum_f|² − Σ_f |direct_f|², 0 )`, where
  ```
  direct_f = bp(holo_f, z)
  illum_f  = asp( bp(holo_f, z_emitter) · emitter_pass,  dz = z_emitter − z,  λ, dx, dy )
  emitter_pass(x,y) = exp( −((x−ex)² + (y−ey)²) / (2σ²) ),   σ = λ_carrier
  ```
  (illumination = emitter focus alone, forward-propagated to each depth.)

---

## 7. Reconstruction B — coherent backprojection (walk scene)  `engine._bp_image / _stack_bp`

Samples: positions `pos_i ∈ ℝ³` (i=0..N−1), complex values `val_{f,i}` per
frequency. Voxel grid `(vx, vy)` at depth `z`. Wavenumber `kw_f = 2π/λ_f`.

**Coherent backprojection** (matched filter for point sources):
```
dist_i(x,y,z) = sqrt( (vx−pos_i.x)² + (vy−pos_i.y)² + (z−pos_i.z)² )
I_f(x,y,z)    = Σ_{i<k}  val_{f,i} · exp( −i · kw_f · dist_i )        (complex)
```

- **single**: `|I_fc|²`.
- **white-light**: `Σ_f |I_f|²`.
- **dark-field** (`subtract_direct`): before summing, remove the known direct
  emitter wave from each sample, then backproject:
  ```
  val'_{f,i} = val_{f,i} − exp( i · kw_f · R_i ) / R_i,     R_i = |pos_i − E|
  ```

---

## 8. Walk field samples  `scenes/walk.build`

Field at an arbitrary 3D antenna position `a` for frequency `f` =
**analytic direct** (exact) + **interpolated cross-scatter**:
```
direct(a)  = exp( i·kw·|a−E| ) / |a−E|
scatter    = bilinear_xy( nearest_z( scatter_planes, a.z ), a.x, a.y )
F(a)       = direct(a) + scatter
```
`scatter_planes[zp] = asp( obj_nomask·mask − obj_nomask, dz = z_object − zp, λ, dx, dy )`,
`obj_nomask = point_source(X, Y, z_object, E, λ)`; planes span the antenna z-range.

Per step `t` (clock-skew phase `u_t = exp(iψ_t)`, `ψ_t ~ U[0,2π)` seeded):
```
S_scan = u_t · F(scan_t),   S_ref = u_t · F(ref_t)
ratio  = S_scan / S_ref     (= F(scan_t)/F(ref_t); clock skew cancels)
val_t  = ratio · F_ref_used
F_ref_used = F(ref_t)   if reference_correction   (pins ref to its true pose)
           = F(ref_0)   otherwise                 (assumes fixed ref -> corrupted)
```
With correction, `val_t = F(scan_t)` (clean emitter-normalized field).

---

## 9. Walk motion (human-like scanning sweep)  `scenes/walk.build`

`frac = linspace(0,1,N)`, `θ = 2π·loops·frac`.
```
env  = sin(π·frac)                         # 0→1→0 : spiral out then in
rad  = 0.12 + 0.88·env
base = ( cx + 0.42·Lx·rad·cos θ,
         cy + 0.42·Ly·rad·sin θ,
         cz + 0.42·walkDepth·sin(2π·1.5·frac) )     # back-and-forth in depth
```
Clumsy hand jitter (low-pass / OU, `σ_j = jitter·0.45`): `j ← 0.85·j + N(0, σ_j)`.
`center_t = reflect( base_t + j_t,  bounds )`  (reflect within walk-mesh box).

Gentle tilt (small smooth wobble, `σ_r = tilt·0.12`):
`angles_t = tilt·(sin(1.3θ+.5), sin(.9θ+1.1), sin(1.7θ)) + N(0, σ_r)`.
Rotation `R = Rz(yaw)·Ry(pitch)·Rx(roll)`.
Antennas: `scan_t = center_t + R_t·offset_scan`, `ref_t = center_t + R_t·offset_ref`.

Reflect(c, [(lo,hi)]): triangle-wave fold `t=(c−lo) mod 2span`, `c=lo + (t if t≤span else 2span−t)`.

---

## 10. Autofocus sharpness (energy-normalized)  `engine.recon`

Per-slice **concentration** metric (brightness-independent), argmax over depth:
```
sharp(z) = (Σ I²) / (Σ I)²          (summed over the slice's pixels)
sharp_index = argmax_z sharp(z)
```
(NOTE: the walk's removed 2nd-peak used **raw** ΣI² for the dark-field curve,
because the energy-normalized version erases an extended object's bump. The live
router/emitter autofocus uses the normalized form above.)

## 11. Focus lateral position  `engine._focus_xy`

Intensity centroid in a `~10 cm` window around the sharpest slice's brightest pixel:
```
(iy, ix) = argmax pixel;  r = round(0.10/dx)
cx = Σ(x·I)/ΣI,  cy = Σ(y·I)/ΣI     over the window, converted to metres
```

## 12. Depth profile & display normalization  `engine.recon / slice_only`

- Profile curve: `prof(z) = max_pixel I(·,·,z) / global_max`.
- Displayed slice: normalized to **its own** max with gamma: `(I / I.max())^0.7`.
- Volume cache: `recon`/`slice_only` reuse the stack for a given `(gen, mode, k)`
  (`gen` bumps on any param change); z-scrubbing then reads the cached slice.

---

## Port checklist (numpy → PyTorch/MPS)

The GPU-worthwhile hot path is **§7 backprojection** (O(samples·voxels·depths)).
To verify a port:
- [ ] §7 `I_f = Σ_i val_i·exp(−i·kw·dist_i)` — on MPS prefer `cos/sin` over complex
      `exp` (compute `real = Σ(vr·cosθ+vi·sinθ)`, `imag = Σ(vi·cosθ−vr·sinθ)`,
      `θ = kw·dist`); then `|I|² = real²+imag²`. Must equal the numpy result to ~1e-5 rel.
- [ ] §7 dark-field subtract-direct applied to `val` before the sum.
- [ ] §10 sharpness and §11 centroid computed on the resulting volume identically.
- [ ] Device fallback: MPS → CPU; identical results (tolerance) on both.
- [ ] Angular-spectrum (§1,§6) may stay numpy (FFT, already fast) or port too;
      if ported, FFT normalization must match numpy's.
