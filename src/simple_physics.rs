//! Pendulum/spring-pendulum physics driving params.

use std::f32::consts::PI;

use bevy::prelude::*;

use crate::{InxParamState, InxPuppet, InxPuppetRoot};

/// Configuration of a SimplePhysics node (immutable after loading).
#[derive(Component, Debug, Clone, Reflect)]
pub struct InxSimplePhysics {
    /// UUID of the param this simulation drives.
    pub param_uuid: u32,
    /// Simulation type.
    pub model: PhysicsModel,
    /// How to map angle/length to the output parameter.
    pub map_mode: PhysicsMapMode,
    /// Gravity for this simulation.
    pub gravity: f32,
    /// Length of the "bone" in pixels.
    pub length: f32,
    /// Oscillation frequency (Hz).
    pub frequency: f32,
    /// Angular damping.
    pub angle_damping: f32,
    /// Length damping.
    pub length_damping: f32,
    /// Output scale (sx, sy).
    pub output_scale: [f32; 2],
    /// If true, physics is relative to the local node, not global.
    pub local_only: bool,
}

/// Simulation model of a `SimplePhysics` node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum PhysicsModel {
    /// Simple pendulum (oscillates under gravity).
    Pendulum,
    /// Spring pendulum (oscillates and extends).
    SpringPendulum,
}

/// How the bob-anchor displacement (XY) maps to the output param's `[f32; 2]` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum PhysicsMapMode {
    /// Angle and length (2D polar).
    AngleLength,
    /// Cartesian X and Y.
    XY,
    /// Length and angle (reverse order).
    LengthAngle,
    /// Cartesian Y and X (reverse order).
    YX,
}

/// Pauses all SimplePhysics simulations when false (global default).
#[derive(Resource, Debug, Clone, Copy, Reflect)]
pub struct PhysicsEnabled(pub bool);

impl Default for PhysicsEnabled {
    fn default() -> Self {
        Self(true)
    }
}

/// Per-puppet override of [`PhysicsEnabled`]. Insert on the puppet's root entity
/// (the `InxPuppetRoot` wrapper) to force physics on/off for that puppet regardless
/// of the global resource - e.g. freeze one puppet's hair sway for a clean capture
/// while another keeps animating.
#[derive(Component, Debug, Clone, Copy, Reflect)]
pub struct InxPuppetPhysicsEnabled(pub bool);

/// Mutable simulation state (per instance).
///
/// All state lives in puppet space (Y downward, same as official Inochi2D): `bob` at
/// rest hangs at `anchor + (0, +length)`. `Pendulum` integrates `(angle, d_angle)`
/// and derives `bob`; `SpringPendulum` integrates `(bob, velocity)` directly.
#[derive(Component, Debug, Default)]
pub struct InxPhysicsState {
    bob: Vec2,
    velocity: Vec2,
    angle: f32,
    d_angle: f32,
    initialized: bool,
}

// Flow: anchor (node's global transform) moves because its parent (head) moves
// - bob experiences inertia (spring-damper)
// - bob-anchor displacement maps to the output param
// - evaluate_params applies the param's bindings to hair/eyes/etc.

/// Spring-damper simulation for every `InxSimplePhysics` node: the anchor (node's own `GlobalTransform`)
/// drags a bob with inertia, and the bob-anchor displacement is written to
/// `InxParamState` as the physics param's value- `evaluate_params` then applies its
/// bindings like any other param (hair sway, accessory jiggle, etc).
pub fn simple_physics_system(
    time: Res<Time>,
    global_enabled: Option<Res<PhysicsEnabled>>,
    mut physics_query: Query<(
        Entity,
        &InxSimplePhysics,
        &mut InxPhysicsState,
        &Transform,
        &GlobalTransform,
    )>,
    puppet_assets: Res<Assets<InxPuppet>>,
    mut root_query: Query<(
        &InxPuppetRoot,
        &mut InxParamState,
        Option<&InxPuppetPhysicsEnabled>,
    )>,
    parents: Query<&ChildOf>,
) {
    // Upstream clamps the whole step to 10 s and subdivides below.
    let dt = time.delta_secs().min(10.0);
    if dt <= 0.0 {
        return;
    }
    let global_enabled = global_enabled.is_none_or(|e| e.0);

    for (entity, config, mut state, transform, gtf) in physics_query.iter_mut() {
        let root = crate::root_of(entity, &parents);
        let Ok((puppet_root, mut param_state, local_override)) = root_query.get_mut(root) else {
            continue;
        };
        // Per-puppet override wins over the global resource.
        let enabled = local_override.map_or(global_enabled, |o| o.0);

        if !enabled {
            state.initialized = false;
            // Write back the neutral offset so the pose returns to default instead
            // of freezing at the last simulated value.
            if param_state.values.get(&config.param_uuid) != Some(&[0.0, 0.0]) {
                param_state.values.insert(config.param_uuid, [0.0, 0.0]);
            }
            continue;
        }

        let (ppm, puppet_gravity) = puppet_assets
            .get(&puppet_root.source)
            .map(|p| (p.physics.pixels_per_meter, p.physics.gravity))
            .unwrap_or((1000.0, 9.8));

        // Upstream: finalGravity = node gravity × puppet gravity × scale.
        let g = config.gravity * puppet_gravity * ppm;

        // `local_only`: anchor tracks the node's own Transform (relative to its immediate parent)
        // instead of GlobalTransform, so the sim isn't dragged by ancestor
        // rotation/scale/scene offset - only the node's own local motion drives it.
        let anchor_up = if config.local_only {
            transform.translation.truncate()
        } else {
            gtf.translation().truncate()
        };
        // The sim runs in puppet space (Y-down, like upstream Inochi2D); Bevy world
        // is Y-up, so flip on the way in and out.
        let anchor = Vec2::new(anchor_up.x, -anchor_up.y);

        if !state.initialized {
            // Bob at rest: hanging below the anchor (+Y in puppet space)
            state.bob = anchor + Vec2::new(0.0, config.length);
            state.velocity = Vec2::ZERO;
            state.angle = 0.0;
            state.d_angle = 0.0;
            state.initialized = true;
            continue;
        }

        // Fixed substeps like upstream: tick(0.01) until exhausted, remainder at the
        // end.
        let mut h = dt;
        while h > 0.01 {
            step(config, &mut state, anchor, 0.01, g);
            h -= 0.01;
        }
        step(config, &mut state, anchor, h, g);

        // Map to param (upstream output convention)
        let (px, py) = map_output(config, &state, anchor, gtf);

        // Write ONLY to the param state of the puppet that owns this physics node.
        // Compare-before-write: keeps InxParamState's change tick quiet once the sim
        // settles, so evaluate_params can skip the puppet.
        if param_state.values.get(&config.param_uuid) != Some(&[px, py]) {
            param_state.values.insert(config.param_uuid, [px, py]);
        }
    }
}

fn step(config: &InxSimplePhysics, state: &mut InxPhysicsState, anchor: Vec2, h: f32, g: f32) {
    // One simulation tick in puppet space (Y-down). Faithful port of Inochi2D v0.8's
    // SimplePhysics (RK4 + continuous critical damping).
    if h <= 0.0 {
        return;
    }
    let l = config.length.max(1.0);

    match config.model {
        PhysicsModel::Pendulum => {
            // ODE angular: θ̈ = −(g/L)·sinθ − θ̇·angleDamping·2√(g/L). The anchor
            // may have moved since the last tick: re-derive θ from the current bob
            // before integrating (so the parent's motion perturbs the pendulum, same as upstream).
            let d = state.bob - anchor;
            if d.length_squared() > 1e-6 {
                state.angle = d.x.atan2(d.y);
            }
            let lr = g / l;
            let crit = 2.0 * lr.max(0.0).sqrt();
            let deriv = |th: f32, w: f32| (w, -lr * th.sin() - w * config.angle_damping * crit);

            let (th0, w0) = (state.angle, state.d_angle);
            let (k1t, k1w) = deriv(th0, w0);
            let (k2t, k2w) = deriv(th0 + k1t * h * 0.5, w0 + k1w * h * 0.5);
            let (k3t, k3w) = deriv(th0 + k2t * h * 0.5, w0 + k2w * h * 0.5);
            let (k4t, k4w) = deriv(th0 + k3t * h, w0 + k3w * h);
            let th = th0 + (h / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
            let w = w0 + (h / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);

            if th.is_finite() && w.is_finite() {
                state.angle = th;
                state.d_angle = w;
            }
            state.bob = anchor + Vec2::new(state.angle.sin(), state.angle.cos()) * l;
        }
        PhysicsModel::SpringPendulum => {
            let k_sqrt = 2.0 * PI * config.frequency;
            let k = (k_sqrt * k_sqrt).max(1e-6);
            // Gravity stretches the rest position: at equilibrium the spring at
            // distance L exactly compensates g.
            let rest_length = l - g / k;
            let lr = g / l;
            let crit_angle = 2.0 * lr.max(0.0).sqrt();
            let crit_length = 2.0 * k_sqrt;

            let deriv = |bob: Vec2, vel: Vec2| -> (Vec2, Vec2) {
                let off = bob - anchor;
                let dist = off.length();
                let n = if dist > 1e-5 { off / dist } else { Vec2::Y };

                let mut force = Vec2::new(0.0, g);
                force -= n * (dist - rest_length) * k;

                // Damping in coordinates rotated to the anchor->bob axis (literal transcription of upstream, cross-terms included).
                let d_rot = Vec2::new(vel.x * n.y + vel.y * n.x, vel.y * n.y - vel.x * n.x);
                let dd_rot = Vec2::new(
                    -d_rot.x * config.angle_damping * crit_angle,
                    -d_rot.y * config.length_damping * crit_length,
                );
                let damp = Vec2::new(
                    dd_rot.x * n.y - d_rot.y * n.x,
                    dd_rot.y * n.y + d_rot.x * n.x,
                );
                (vel, force + damp)
            };

            let (b0, v0) = (state.bob, state.velocity);
            let (k1b, k1v) = deriv(b0, v0);
            let (k2b, k2v) = deriv(b0 + k1b * h * 0.5, v0 + k1v * h * 0.5);
            let (k3b, k3v) = deriv(b0 + k2b * h * 0.5, v0 + k2v * h * 0.5);
            let (k4b, k4v) = deriv(b0 + k3b * h, v0 + k3v * h);
            let b = b0 + (k1b + k2b * 2.0 + k3b * 2.0 + k4b) * (h / 6.0);
            let v = v0 + (k1v + k2v * 2.0 + k3v * 2.0 + k4v) * (h / 6.0);

            // Upstream sanity check: if anything goes to NaN/inf, revert.
            if b.is_finite() && v.is_finite() {
                state.bob = b;
                state.velocity = v;
            }
        }
    }
}

/// Maps the sim output to the param value (upstream convention: normalized direction in the node's local space, angle/π, length relative to `length`).
fn map_output(
    config: &InxSimplePhysics,
    state: &InxPhysicsState,
    anchor: Vec2,
    gtf: &GlobalTransform,
) -> (f32, f32) {
    let l = config.length.max(1.0);
    let sx = config.output_scale[0];
    let sy = config.output_scale[1];

    // Output direction in the node's local space (upstream applies the inverse of the world matrix unless localOnly),
    // converted to Y-down.
    let local = if config.local_only {
        state.bob - anchor
    } else {
        let out_up = Vec3::new(state.bob.x, -state.bob.y, 0.0);
        let p = gtf.to_matrix().inverse().transform_point3(out_up);
        Vec2::new(p.x, -p.y)
    };
    let local_angle = local.normalize_or(Vec2::Y);
    let rel_length = state.bob.distance(anchor) / l;

    let (px, py) = match config.map_mode {
        PhysicsMapMode::AngleLength => {
            let a = (-local_angle.x).atan2(local_angle.y) / PI;
            (a, rel_length)
        }
        PhysicsMapMode::LengthAngle => {
            let a = (-local_angle.x).atan2(local_angle.y) / PI;
            (rel_length, a)
        }
        PhysicsMapMode::XY => {
            let p = local_angle * rel_length - Vec2::new(0.0, 1.0);
            (p.x, -p.y)
        }
        PhysicsMapMode::YX => {
            let p = local_angle * rel_length - Vec2::new(0.0, 1.0);
            (-p.y, p.x)
        }
    };
    (px * sx, py * sy)
}

#[cfg(feature = "inx")]
impl From<inochi2d_parser::prelude::PhysicsModelType> for PhysicsModel {
    fn from(m: inochi2d_parser::prelude::PhysicsModelType) -> Self {
        match m {
            inochi2d_parser::prelude::PhysicsModelType::Pendulum => Self::Pendulum,
            inochi2d_parser::prelude::PhysicsModelType::SpringPendulum => Self::SpringPendulum,
        }
    }
}

#[cfg(feature = "inx")]
impl From<inochi2d_parser::prelude::PhysicsMapMode> for PhysicsMapMode {
    fn from(m: inochi2d_parser::prelude::PhysicsMapMode) -> Self {
        match m {
            inochi2d_parser::prelude::PhysicsMapMode::AngleLength => Self::AngleLength,
            inochi2d_parser::prelude::PhysicsMapMode::XY => Self::XY,
            inochi2d_parser::prelude::PhysicsMapMode::LengthAngle => Self::LengthAngle,
            inochi2d_parser::prelude::PhysicsMapMode::YX => Self::YX,
        }
    }
}
