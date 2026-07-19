//! 2D grid interpolation (linear/cubic/stepped) shared by params and animations.

use crate::*;

/// Interpolates an f32 value from a binding's grid, respecting is_set and the interpolation mode.
pub fn interpolate_binding(param: &InxParam, value: &[f32; 2], binding: &InxBinding) -> f32 {
    let InxBindingValues::Transform(flat) = &binding.values else {
        return 0.0;
    };

    if flat.frames == 0 || flat.values_per_frame == 0 || flat.data.is_empty() {
        return 0.0;
    }

    let (nx, ny) = normalize_param(param, value);
    let ax = &param.axis_points[0];
    let ay = &param.axis_points[1];

    if param.is_vec2 && ay.len() > 1 {
        // 2D grid
        let (ix, fx) = find_grid_pos(ax, nx);
        let (iy, fy) = find_grid_pos(ay, ny);
        let max_x = flat.frames.saturating_sub(1);
        let max_y = flat.values_per_frame.saturating_sub(1);

        let get =
            |x: usize, y: usize| -> f32 { flat.get(x.min(max_x), y.min(max_y)).unwrap_or(0.0) };

        // Resolve is_set: if any corner isn't set, use the value of the nearest set
        // corner
        let corners = resolve_is_set_scalar(&binding.is_set, ix, iy, max_x, max_y, &get);

        interpolate_2d(corners, fx, fy, binding.interpolation)
    } else {
        // 1D grid
        let (ix, fx) = find_grid_pos(ax, nx);
        let max_x = flat.frames.saturating_sub(1);

        let get = |x: usize| -> f32 { flat.get(x.min(max_x), 0).unwrap_or(0.0) };

        let v0 = resolve_1d_value(&binding.is_set, ix, max_x, &get);
        let v1 = resolve_1d_value(&binding.is_set, (ix + 1).min(max_x), max_x, &get);

        match binding.interpolation {
            InxInterpolation::Stepped => v0,
            InxInterpolation::Linear => lerp(v0, v1, fx),
            InxInterpolation::Cubic => {
                let v_prev = if ix > 0 {
                    resolve_1d_value(&binding.is_set, ix - 1, max_x, &get)
                } else {
                    v0
                };
                let v_next2 = resolve_1d_value(&binding.is_set, (ix + 2).min(max_x), max_x, &get);
                catmull_rom(v_prev, v0, v1, v_next2, fx)
            }
        }
    }
}

/// Interpolates deformation (offsets per vertex) from a binding's grid.
pub fn interpolate_binding_deform(
    param: &InxParam,
    value: &[f32; 2],
    binding: &InxBinding,
) -> Option<Vec<[f32; 2]>> {
    let InxBindingValues::Deform(flat) = &binding.values else {
        return None;
    };

    if flat.data.is_empty() || flat.frames == 0 || flat.vertices_per_frame == 0 {
        return None;
    }

    let (nx, ny) = normalize_param(param, value);
    let ax = &param.axis_points[0];
    let ay = &param.axis_points[1];
    let vpf = flat.vertices_per_frame;
    let y_frames = ay.len();

    if param.is_vec2 && y_frames > 1 {
        let real_vpf = vpf / y_frames;
        let (ix, fx) = find_grid_pos(ax, nx);
        let (iy, fy) = find_grid_pos(ay, ny);

        let max_x = ax.len().saturating_sub(1);
        let max_y = y_frames.saturating_sub(1);

        let get_vertex = |xi: usize, yi: usize, vi: usize| -> [f32; 2] {
            let xi = xi.min(max_x);
            let yi = yi.min(max_y);
            let idx = xi * vpf + yi * real_vpf + vi;
            flat.data.get(idx).copied().unwrap_or([0.0, 0.0])
        };

        let mut result = vec![[0.0f32; 2]; real_vpf];
        for (vi, slot) in result.iter_mut().enumerate() {
            let v00 = get_vertex(ix, iy, vi);
            let v10 = get_vertex(ix + 1, iy, vi);
            let v01 = get_vertex(ix, iy + 1, vi);
            let v11 = get_vertex(ix + 1, iy + 1, vi);

            *slot = interpolate_2d_vec2([v00, v10, v01, v11], fx, fy, binding.interpolation);
            slot[1] = -slot[1]; // Y inverted
        }

        Some(result)
    } else {
        let (ix, fx) = find_grid_pos(ax, nx);

        let get_vertex = |xi: usize, yi: usize, vi: usize| -> [f32; 2] {
            let xi = xi.min(ax.len().saturating_sub(1));
            let yi = yi.min(y_frames.saturating_sub(1));
            let idx = (xi * y_frames + yi) * vpf + vi;
            flat.data.get(idx).copied().unwrap_or([0.0, 0.0])
        };

        let mut result = vec![[0.0f32; 2]; vpf];
        for (vi, slot) in result.iter_mut().enumerate() {
            let v0 = get_vertex(ix, 0, vi);
            let v1 = get_vertex(ix + 1, 0, vi);

            *slot = match binding.interpolation {
                InxInterpolation::Stepped => [v0[0], -v0[1]],
                InxInterpolation::Linear | InxInterpolation::Cubic => {
                    [lerp(v0[0], v1[0], fx), -(lerp(v0[1], v1[1], fx))]
                }
            };
        }
        Some(result)
    }
}

// 2D INTERPOLATION BY MODE

/// Corners: [v00, v10, v01, v11]
fn interpolate_2d(corners: [f32; 4], fx: f32, fy: f32, mode: InxInterpolation) -> f32 {
    let [v00, v10, v01, v11] = corners;

    match mode {
        InxInterpolation::Stepped => {
            // Snap to the nearest corner
            let xi = if fx < 0.5 { 0 } else { 1 };
            let yi = if fy < 0.5 { 0 } else { 1 };
            match (xi, yi) {
                (0, 0) => v00,
                (1, 0) => v10,
                (0, 1) => v01,
                _ => v11,
            }
        }
        InxInterpolation::Linear => {
            let top = lerp(v00, v10, fx);
            let bot = lerp(v01, v11, fx);
            lerp(top, bot, fy)
        }
        InxInterpolation::Cubic => {
            // Cubic 2D would need 4x4 neighbors. Fallback to bilinear (enough for most models).
            let top = lerp(v00, v10, fx);
            let bot = lerp(v01, v11, fx);
            lerp(top, bot, fy)
        }
    }
}

fn interpolate_2d_vec2(
    corners: [[f32; 2]; 4],
    fx: f32,
    fy: f32,
    mode: InxInterpolation,
) -> [f32; 2] {
    let cx = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]];
    let cy = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]];
    [
        interpolate_2d(cx, fx, fy, mode),
        interpolate_2d(cy, fx, fy, mode),
    ]
}

// IS_SET RESOLUTION

/// For a 2D cell, resolves the 4 corners. If any isn't set, looks for the nearest
/// set corner. Returns [v00, v10, v01, v11].
fn resolve_is_set_scalar<F>(
    is_set: &[Vec<bool>],
    ix: usize,
    iy: usize,
    max_x: usize,
    max_y: usize,
    get: &F,
) -> [f32; 4]
where
    F: Fn(usize, usize) -> f32,
{
    let x0 = ix.min(max_x);
    let x1 = (ix + 1).min(max_x);
    let y0 = iy.min(max_y);
    let y1 = (iy + 1).min(max_y);

    if is_set.is_empty() {
        // No mask -> everything is set
        return [get(x0, y0), get(x1, y0), get(x0, y1), get(x1, y1)];
    }

    let check = |x: usize, y: usize| -> bool {
        is_set
            .get(x)
            .and_then(|row| row.get(y))
            .copied()
            .unwrap_or(true) // out of range = consider it set
    };

    let resolve = |x: usize, y: usize| -> f32 {
        if check(x, y) {
            return get(x, y);
        }
        // Look for the nearest set cell (simple spiral)
        find_nearest_set_value(is_set, x, y, max_x, max_y, get)
    };

    [
        resolve(x0, y0),
        resolve(x1, y0),
        resolve(x0, y1),
        resolve(x1, y1),
    ]
}

/// Looks for the value of the nearest set cell to (cx, cy).
fn find_nearest_set_value<F>(
    is_set: &[Vec<bool>],
    cx: usize,
    cy: usize,
    max_x: usize,
    max_y: usize,
    get: &F,
) -> f32
where
    F: Fn(usize, usize) -> f32,
{
    let mut best_dist = u32::MAX;
    let mut best_val = 0.0f32;

    for x in 0..=max_x {
        for y in 0..=max_y {
            let is = is_set
                .get(x)
                .and_then(|row| row.get(y))
                .copied()
                .unwrap_or(false);
            if !is {
                continue;
            }
            let dx = x.abs_diff(cx) as u32;
            let dy = y.abs_diff(cy) as u32;
            let dist = dx * dx + dy * dy;
            if dist < best_dist {
                best_dist = dist;
                best_val = get(x, y);
            }
        }
    }

    best_val
}

/// For 1D: resolve is_set for an index.
fn resolve_1d_value<F>(is_set: &[Vec<bool>], x: usize, max_x: usize, get: &F) -> f32
where
    F: Fn(usize) -> f32,
{
    if is_set.is_empty() {
        return get(x);
    }
    // 1D: is_set[x][0]
    let is = is_set
        .get(x)
        .and_then(|row| row.first())
        .copied()
        .unwrap_or(true);
    if is {
        return get(x);
    }
    // Look for the nearest neighbor that's set
    for dist in 1..=max_x {
        if x >= dist {
            let xl = x - dist;
            let is_l = is_set
                .get(xl)
                .and_then(|row| row.first())
                .copied()
                .unwrap_or(false);
            if is_l {
                return get(xl);
            }
        }
        let xr = x + dist;
        if xr <= max_x {
            let is_r = is_set
                .get(xr)
                .and_then(|row| row.first())
                .copied()
                .unwrap_or(false);
            if is_r {
                return get(xr);
            }
        }
    }
    get(x) // fallback
}

// PARAM MERGE MODE - for evaluate_params

/// Deform accumulator per node UUID.
#[derive(Default)]
pub struct DeformAccum {
    /// Accumulated [dx, dy] offset per vertex.
    pub offsets: Vec<[f32; 2]>,
    /// True once at least one binding has written into `offsets`.
    pub _initialized: bool,
}

/// Per-node accumulator that respects the param's merge_mode.
pub struct NodeAccum {
    /// Accumulated translation X offset.
    pub tx: f32,
    /// Accumulated translation Y offset.
    pub ty: f32,
    /// Accumulated translation Z offset.
    pub tz: f32,
    /// Accumulated scale X factor.
    pub sx: f32,
    /// Accumulated scale Y factor.
    pub sy: f32,
    /// Accumulated rotation X offset.
    pub rx: f32,
    /// Accumulated rotation Y offset.
    pub ry: f32,
    /// Accumulated rotation Z offset.
    pub rz: f32,
    /// Accumulated opacity value.
    pub opacity: f32,
    /// True once at least one binding has written `opacity`.
    pub has_opacity: bool,
}

impl Default for NodeAccum {
    fn default() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            tz: 0.0,
            sx: 1.0,
            sy: 1.0,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            opacity: 1.0,
            has_opacity: false,
        }
    }
}

impl NodeAccum {
    /// Accumulates a binding value according to the param's merge_mode.
    pub fn accumulate(&mut self, param_name: &InxParamName, value: f32, merge_mode: InxMergeMode) {
        match param_name {
            InxParamName::TransformTX => merge_field(&mut self.tx, value, merge_mode),
            InxParamName::TransformTY => merge_field(&mut self.ty, value, merge_mode),
            InxParamName::TransformTZ => merge_field(&mut self.tz, value, merge_mode),
            InxParamName::TransformSX => merge_scale(&mut self.sx, value, merge_mode),
            InxParamName::TransformSY => merge_scale(&mut self.sy, value, merge_mode),
            InxParamName::TransformRX => merge_field(&mut self.rx, value, merge_mode),
            InxParamName::TransformRY => merge_field(&mut self.ry, value, merge_mode),
            InxParamName::TransformRZ => merge_field(&mut self.rz, value, merge_mode),
            InxParamName::Opacity => {
                self.opacity = value;
                self.has_opacity = true;
            }
            _ => {}
        }
    }
}

fn merge_field(field: &mut f32, value: f32, mode: InxMergeMode) {
    match mode {
        InxMergeMode::Additive => *field += value,
        InxMergeMode::Multiply => *field *= value,
        InxMergeMode::Override | InxMergeMode::Forced => *field = value,
    }
}

fn merge_scale(field: &mut f32, value: f32, mode: InxMergeMode) {
    match mode {
        InxMergeMode::Additive => {
            // For additive scale: accumulate as multiplication because scale's
            // "default" is 1.0, not 0.0
            *field *= value;
        }
        InxMergeMode::Multiply => *field *= value,
        InxMergeMode::Override | InxMergeMode::Forced => *field = value,
    }
}

// UTILITIES

fn normalize_param(param: &InxParam, value: &[f32; 2]) -> (f32, f32) {
    let rx = param.max[0] - param.min[0];
    let ry = param.max[1] - param.min[1];
    let nx = if rx.abs() > f32::EPSILON {
        ((value[0] - param.min[0]) / rx).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ny = if ry.abs() > f32::EPSILON {
        ((value[1] - param.min[1]) / ry).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (nx, ny)
}

/// Locates `normalized` (0..1) within a param's non-uniform `axis_points`: returns
/// the segment index and the fractional position within it.
pub fn find_grid_pos(axis_points: &[f32], normalized: f32) -> (usize, f32) {
    if axis_points.len() <= 1 {
        return (0, 0.0);
    }
    for i in 0..axis_points.len() - 1 {
        let a = axis_points[i];
        let b = axis_points[i + 1];
        if normalized <= b || i == axis_points.len() - 2 {
            let range = b - a;
            let frac = if range.abs() > f32::EPSILON {
                ((normalized - a) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
            return (i, frac);
        }
    }
    (axis_points.len() - 2, 1.0)
}

/// Linear interpolation.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite spline between two keyframes, `tension` in `[0, 1]` (0 = full Catmull-Rom-like curve, 1 = linear).
#[inline]
pub fn cubic_hermite(a: f32, b: f32, t: f32, tension: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let s = 1.0 - tension;
    let h1 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h2 = -2.0 * t3 + 3.0 * t2;
    let h3 = t3 - 2.0 * t2 + t;
    let h4 = t3 - t2;
    let tangent = s * (b - a);
    h1 * a + h2 * b + h3 * tangent + h4 * tangent
}

/// Ease in-out (smoothstep) for smooth transitions.
#[inline]
pub fn ease_in_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}


/// Catmull-Rom 1D. p0..p3 are 4 points, t ∈ `[0,1]` interpolates between p1 and p2.
#[inline]
pub fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}
