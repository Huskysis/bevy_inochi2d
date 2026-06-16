//! Pendulum / spring-pendulum simulation for SimplePhysics nodes.
//!
//! Integrates the configured [`PhysicsModel`] each frame using the puppet's
//! gravity and pixels-per-meter, then writes the result back into the
//! linked param according to [`PhysicsMapMode`] so it feeds the regular
//! binding pipeline.

use std::f32::consts::PI;

use bevy::prelude::*;

use crate::{InxParamState, InxPuppet, InxPuppetRoot};

/// Configuracion del nodo SimplePhysics (inmutable post-carga).
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

/// Estado mutable de la simulacion (por instancia).
#[derive(Component, Debug, Default)]
pub struct InxPhysicsState {
    bob: Vec2,
    velocity: Vec2,
    prev_anchor: Vec2,
    initialized: bool,
}

// Flujo:
// anchor (globalt del nodo) se mueve porque su padre (head) se mueve
// - bob experimenta inercia (spring-damper)
// - displacement bob-anchor se mapea al param de salida
// - evaluate_params aplica los bindings del param al pelo/ojos/etc.

pub fn simple_physics_system(
    time: Res<Time>,
    mut physics_query: Query<(&InxSimplePhysics, &mut InxPhysicsState, &GlobalTransform)>,
    puppet_assets: Res<Assets<InxPuppet>>,
    mut root_query: Query<(&InxPuppetRoot, &mut InxParamState)>,
) {
    let dt = time.delta_secs().min(0.05);
    if dt <= 0.0 {
        return;
    }

    // Obtener ppm del primer puppet (TODO: multi-puppet)
    let ppm = root_query
        .iter()
        .next()
        .and_then(|(root, _)| puppet_assets.get(&root.source))
        .map(|p| p.physics.pixels_per_meter)
        .unwrap_or(1000.0);

    for (config, mut state, gtf) in physics_query.iter_mut() {
        let anchor = gtf.translation().truncate();

        if !state.initialized {
            // Bob en reposo: directamente debajo del anchor
            // En Bevy (+Y = arriba), "debajo" = Y negativo
            state.bob = anchor + Vec2::new(0.0, -config.length);
            state.prev_anchor = anchor;
            state.velocity = Vec2::ZERO;
            state.initialized = true;
            continue;
        }

        // Simular
        simulate(config, &mut state, anchor, dt, ppm);

        // Mapear a param
        let diff = state.bob - anchor;
        let (px, py) = map_output(config, diff);

        // Escribir al param state del puppet
        // Si la animacion ya escribio este param (Forced), no "pisar".
        // Detectar: si el entry ya existe despues del clear() de animation,
        // significa que animation lo escribio = respetar.
        for (_root, mut param_state) in root_query.iter_mut() {
            // if param_state.values.contains_key(&config.param_uuid) {
            //     // Animacion tiene prioridad (Forced)
            //     continue;
            // }
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
    let k = (2.0 * PI * config.frequency).powi(2);

    // Posicion de reposo (colgando debajo)
    let rest = anchor + Vec2::new(0.0, -config.length);

    // Spring: tira el bob hacia rest
    let spring = (rest - state.bob) * k;

    // Gravedad: hacia abajo (Y negativo en Bevy)
    let gravity = Vec2::new(0.0, config.gravity * ppm);

    // Integrar velocidad
    state.velocity += (spring + gravity) * dt;

    // Amortiguacion en coordenadas polares respecto al anchor
    let to_bob = state.bob - anchor;
    let dist = to_bob.length();

    if dist > 0.001 {
        let radial = to_bob / dist;
        let tangent = Vec2::new(-radial.y, radial.x);

        let vr = state.velocity.dot(radial) * (1.0 - config.length_damping);
        let vt = state.velocity.dot(tangent) * (1.0 - config.angle_damping);

        state.velocity = radial * vr + tangent * vt;
    }

    // Integrar posicion
    state.bob += state.velocity * dt;

    // Pendulum: constraint de longitud fija
    if config.model == PhysicsModel::Pendulum {
        let to_bob = state.bob - anchor;
        let d = to_bob.length();
        if d > 0.001 {
            let dir = to_bob / d;
            state.bob = anchor + dir * config.length;
            // Eliminar velocidad radial
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
            // Angulo desde la vertical (reposo = straight down = -Y)
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
            // Normalizado por longitud.
            // diff.y en reposo = -length => (-(-len)/len - 1.0) = 0
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
