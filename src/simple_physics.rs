//! Pendulum / spring-pendulum simulation for SimplePhysics nodes.
//!
//! Integrates the configured [`PhysicsModel`] each frame using the puppet's
//! gravity and pixels-per-meter, then writes the result back into the
//! linked param according to [`PhysicsMapMode`] so it feeds the regular
//! binding pipeline.

use std::f32::consts::PI;

use bevy::prelude::*;

use crate::{InxAnimationController, InxParamState, InxPuppet, InxPuppetRoot};

/// SimplePhysics node config (immutable after load).
#[derive(Component, Debug, Clone, Reflect)]
pub struct InxSimplePhysics {
    pub param_uuid: u32,
    pub model: PhysicsModel,
    pub map_mode: PhysicsMapMode,
    pub gravity: f32,
    pub length: f32,
    pub frequency: f32,
    pub angle_damping: f32,
    pub length_damping: f32,
    pub output_scale: [f32; 2],
    pub local_only: bool,
    /// Pauses this physics node. State is reset on resume.
    pub enabled: bool,
}

/// Pauses all SimplePhysics simulations when false.
#[derive(Resource, Debug, Clone, Copy, Reflect)]
pub struct PhysicsEnabled(pub bool);

impl Default for PhysicsEnabled {
    fn default() -> Self {
        Self(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum PhysicsModel {
    Pendulum,
    SpringPendulum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum PhysicsMapMode {
    AngleLength,
    XY,
    LengthAngle,
    YX,
}

/// Per-instance mutable simulation state.
#[derive(Component, Debug, Default, Reflect)]
pub struct InxPhysicsState {
    bob: Vec2,
    velocity: Vec2,
    prev_anchor: Vec2,
    initialized: bool,
}

// Flow:
// - anchor (node's global transform) moves because its parent (e.g. head) moves
// - bob lags behind via spring-damper inertia
// - (bob - anchor) is mapped to the output param
// - evaluate_params then applies the param bindings to hair/eyes/etc.
pub fn simple_physics_system(
    time: Res<Time>,
    enabled: Option<Res<PhysicsEnabled>>,
    mut physics_query: Query<(&InxSimplePhysics, &mut InxPhysicsState, &GlobalTransform)>,
    puppet_assets: Res<Assets<InxPuppet>>,
    mut root_query: Query<(&InxPuppetRoot, &mut InxParamState, Option<&InxAnimationController>)>,
) {
    let dt = time.delta_secs().min(0.05);
    if dt <= 0.0 {
        return;
    }

    let global_on = enabled.map_or(true, |e| e.0);

    // Pull pixels-per-meter from the first puppet (TODO: multi-puppet).
    let ppm = root_query
        .iter()
        .next()
        .and_then(|(root, _, _)| puppet_assets.get(&root.source))
        .map(|p| p.physics.pixels_per_meter)
        .unwrap_or(1000.0);

    for (config, mut state, gtf) in physics_query.iter_mut() {
        // Global or per-node pause: reset state, skip param write.
        // Param falls back to default via animation's `param_defaults`.
        if !global_on || !config.enabled {
            state.initialized = false;
            continue;
        }

        let anchor = gtf.translation().truncate();

        if !state.initialized {
            // Rest pose: bob hangs straight below the anchor.
            // Bevy uses +Y up, so "below" means -Y.
            state.bob = anchor + Vec2::new(0.0, -config.length);
            state.prev_anchor = anchor;
            state.velocity = Vec2::ZERO;
            state.initialized = true;
            continue;
        }

        simulate(config, &mut state, anchor, dt, ppm);

        let diff = state.bob - anchor;
        let (px, py) = map_output(config, diff);

        for (_root, mut param_state, _controller) in root_query.iter_mut() {
            param_state.values.insert(config.param_uuid, [px, py]);
        }

        state.prev_anchor = anchor;
    }
}

fn simulate(
    config: &InxSimplePhysics,
    state: &mut InxPhysicsState,
    anchor: Vec2,
    dt: f32,
    ppm: f32,
) {
    // Natural angular frequency of the spring (rad/s).
    let omega = 2.0 * PI * config.frequency;
    let k = omega * omega;

    // Rest position: straight below the anchor.
    let rest = anchor + Vec2::new(0.0, -config.length);

    // Spring force pulls bob toward rest.
    let spring = (rest - state.bob) * k;

    // Gravity (sign matches puppet convention; inverted by the loader if needed).
    let gravity = Vec2::new(0.0, -config.gravity * ppm);

    // Semi-implicit Euler: integrate velocity first, then position.
    state.velocity += (spring + gravity) * dt;

    // Damping in polar coords relative to the anchor.
    // Use exponential decay tied to natural frequency: critically damped at
    // ratio = 1. Stable for any dt and consistent with a damped harmonic
    // oscillator, instead of the framerate-dependent (1 - damping) per frame.
    let to_bob = state.bob - anchor;
    let dist = to_bob.length();

    if dist > 0.001 {
        let radial = to_bob / dist;
        let tangent = Vec2::new(-radial.y, radial.x);

        let decay_r = (-2.0 * config.length_damping * omega * dt).exp();
        let decay_t = (-2.0 * config.angle_damping * omega * dt).exp();

        let vr = state.velocity.dot(radial) * decay_r;
        let vt = state.velocity.dot(tangent) * decay_t;

        state.velocity = radial * vr + tangent * vt;
    }

    // Integrate position with the (now damped) velocity.
    state.bob += state.velocity * dt;

    // Pendulum model: hard length constraint.
    if config.model == PhysicsModel::Pendulum {
        let to_bob = state.bob - anchor;
        let d = to_bob.length();
        if d > 0.001 {
            let dir = to_bob / d;
            state.bob = anchor + dir * config.length;
            // Project out radial velocity component.
            state.velocity -= dir * state.velocity.dot(dir);
        }
    }
}

fn map_output(config: &InxSimplePhysics, diff: Vec2) -> (f32, f32) {
    let len = config.length.max(1.0);
    let sx = config.output_scale[0];
    let sy = config.output_scale[1];

    match config.map_mode {
        PhysicsMapMode::AngleLength => {
            // Angle from vertical (rest = straight down = -Y).
            let angle = diff.x.atan2(-diff.y);
            let extension = (diff.length() - len) / len;
            (angle * sx, extension * sy)
        }
        PhysicsMapMode::LengthAngle => {
            let angle = diff.x.atan2(-diff.y);
            let extension = (diff.length() - len) / len;
            (extension * sx, angle * sy)
        }
        PhysicsMapMode::XY => {
            // Normalized by rest length.
            // At rest diff.y = -length, so ny = -(-len)/len - 1 = 0.
            let nx = diff.x / len;
            let ny = -diff.y / len - 1.0;
            (nx * sx, ny * sy)
        }
        PhysicsMapMode::YX => {
            let nx = diff.x / len;
            let ny = -diff.y / len - 1.0;
            (ny * sx, nx * sy)
        }
    }
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
