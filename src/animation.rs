//! Multi-layer animation playback for puppets.
//!
//! Ticks each [`AnimationLayer`] on an [`InxAnimationController`], advances
//! fade-in/out states, samples [`InxAnimationLane`] keyframes and writes the
//! blended result into [`InxParamState`] for the param-evaluation pass.

use bevy::{platform::collections::HashMap, prelude::*};

use crate::{
    FadeState, InxAnimation, InxAnimationController, InxAnimationLane, InxBasePose, InxBinding,
    InxBindingValues, InxDeform, InxInterpolation, InxMaterial, InxMergeMode, InxParam,
    InxParamState, InxPuppet, InxPuppetRoot, InxUUID,
    grid_interpolation::{DeformAccum, NodeAccum, cubic_hermite, ease_in_out, lerp},
};

/// Avanza tiempo de cada capa, actualiza fades, evalúa lanes,
/// y blendea los resultados ponderados en InxParamState.
pub fn update_animation_controller(
    time: Res<Time>,
    animations: Res<Assets<InxAnimation>>,
    mut query: Query<(&mut InxAnimationController, &mut InxParamState)>,
) {
    let dt = time.delta_secs();

    for (mut controller, mut state) in query.iter_mut() {
        // tick capas (tiempo + fade)
        for layer in controller.layers.iter_mut() {
            if !layer.playing {
                continue;
            }

            // Avanzar tiempo
            layer.time += dt * layer.speed;

            if let Some(anim) = animations.get(&layer.animation)
                && layer.time >= anim.duration
            {
                if layer.looping {
                    layer.time %= anim.duration;
                } else {
                    layer.time = anim.duration;
                    layer.playing = false;
                }
            }

            // Actualizar fade
            match &mut layer.fade {
                FadeState::None => {}
                FadeState::FadingIn { duration, elapsed } => {
                    *elapsed += dt;
                    let t = (*elapsed / *duration).min(1.0);
                    layer.weight = ease_in_out(t);
                    if t >= 1.0 {
                        layer.weight = 1.0;
                        layer.fade = FadeState::None;
                    }
                }
                FadeState::FadingOut {
                    duration,
                    elapsed,
                    start_weight,
                } => {
                    *elapsed += dt;
                    let t = (*elapsed / *duration).min(1.0);
                    layer.weight = *start_weight * (1.0 - ease_in_out(t));
                    if t >= 1.0 {
                        layer.weight = 0.0;
                        layer.playing = false;
                        layer.fade = FadeState::None;
                    }
                }
            }
        }

        // Limpiar capas muertas (weight 0 + no playing), excepto layer 0 (idle)
        let layer_count = controller.layers.len();
        if layer_count > 1 {
            controller
                .layers
                .retain_mut(|l| l.playing || l.weight > 0.001);
            // Asegurar que siempre quede al menos una
            if controller.layers.is_empty() {
                // No debería pasar, pero por seguridad
            }
        }

        let mut layer_values: Vec<(f32, HashMap<u32, [f32; 2]>)> = Vec::new();

        for layer in controller.layers.iter() {
            if layer.weight < 0.001 || !layer.playing {
                continue;
            }

            let Some(anim) = animations.get(&layer.animation) else {
                continue;
            };

            let frame = if anim.timestep > 0.0 {
                layer.time / anim.timestep
            } else {
                0.0
            };

            let mut values: HashMap<u32, [f32; 2]> = HashMap::default();

            for lane in &anim.lanes {
                let value = evaluate_lane(lane, frame);
                let entry = values.entry(lane.param_uuid).or_insert([0.0, 0.0]);
                // Lanes dentro de la misma animación aplican merge_mode normal
                match lane.merge_mode {
                    InxMergeMode::Additive => entry[lane.target as usize] += value,
                    InxMergeMode::Multiply => entry[lane.target as usize] *= value,
                    InxMergeMode::Override | InxMergeMode::Forced => {
                        entry[lane.target as usize] = value;
                    }
                }
            }

            layer_values.push((layer.weight, values));
        }

        // Reset params animables a defaults; physics-only uuids (ausentes en
        // param_defaults) quedan intactos en state.values.
        for (&uuid, &default) in &controller.param_defaults {
            state.values.insert(uuid, default);
        }

        // Blend capas activas encima de los defaults.
        for (weight, values) in &layer_values {
            for (&uuid, &layer_val) in values {
                let current = state.values.entry(uuid).or_insert([0.0, 0.0]);
                current[0] = lerp(current[0], layer_val[0], *weight);
                current[1] = lerp(current[1], layer_val[1], *weight);
            }
        }
    }
}

fn evaluate_lane(lane: &InxAnimationLane, frame: f32) -> f32 {
    let kfs = &lane.keyframes;
    if kfs.is_empty() {
        return 0.0;
    }
    if kfs.len() == 1 || frame <= kfs[0].frame as f32 {
        return kfs[0].value;
    }
    let last = kfs.last().unwrap();
    if frame >= last.frame as f32 {
        return last.value;
    }

    let mut prev_idx = 0;
    for (i, kf) in kfs.iter().enumerate() {
        if kf.frame as f32 > frame {
            break;
        }
        prev_idx = i;
    }

    let prev = &kfs[prev_idx];
    let next = &kfs[prev_idx + 1];
    let span = next.frame as f32 - prev.frame as f32;
    if span <= 0.0 {
        return prev.value;
    }
    let t = (frame - prev.frame as f32) / span;

    match lane.interpolation {
        InxInterpolation::Stepped => prev.value,
        InxInterpolation::Linear => lerp(prev.value, next.value, t),
        InxInterpolation::Cubic => {
            let tension = (prev.tension + next.tension) * 0.5;
            cubic_hermite(prev.value, next.value, t, tension)
        }
    }
}

pub fn evaluate_params(
    puppet_query: Query<(&InxParamState, &InxPuppetRoot)>,
    puppet_assets: Res<Assets<InxPuppet>>,
    param_assets: Res<Assets<InxParam>>,
    mut node_query: Query<(
        &InxUUID,
        &InxBasePose,
        &mut Transform,
        Option<&mut InxMaterial>,
        Option<&mut InxDeform>,
    )>,
) {
    for (state, root) in puppet_query.iter() {
        let Some(puppet) = puppet_assets.get(&root.source) else {
            continue;
        };

        // acumuladores
        let mut transform_accum: HashMap<u32, NodeAccum> = HashMap::default();
        let mut deform_accum: HashMap<u32, DeformAccum> = HashMap::default();

        for param_handle in &puppet.params {
            let Some(param) = param_assets.get(param_handle) else {
                continue;
            };

            let param_value = state
                .values
                .get(&param.uuid)
                .copied()
                .unwrap_or(param.defaults);

            for binding in &param.bindings {
                match &binding.values {
                    InxBindingValues::Transform(_) => {
                        let value = interpolate_binding(param, &param_value, binding);
                        let entry = transform_accum.entry(binding.node_uuid).or_default();

                        entry.accumulate(&binding.param_name, value, param.merge_mode);
                    }
                    InxBindingValues::Deform(_) => {
                        if let Some(offsets) =
                            interpolate_binding_deform(param, &param_value, binding)
                        {
                            let entry =
                                deform_accum.entry(binding.node_uuid).or_insert_with(|| {
                                    DeformAccum {
                                        offsets: vec![[0.0; 2]; offsets.len()],
                                        _initialized: false,
                                    }
                                });
                            if entry.offsets.len() == offsets.len() {
                                for (acc, src) in entry.offsets.iter_mut().zip(offsets.iter()) {
                                    acc[0] += src[0];
                                    acc[1] += src[1];
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        for (uuid, base, mut transform, material, deform) in node_query.iter_mut() {
            if let Some(offsets) = transform_accum.get(&uuid.0) {
                transform.translation.x = base.translation.x + offsets.tx;
                transform.translation.y = base.translation.y - offsets.ty;
                transform.translation.z = base.translation.z + offsets.tz;

                transform.scale.x = base.scale.x * offsets.sx;
                transform.scale.y = base.scale.y * offsets.sy;

                if offsets.rz.abs() > 1e-6 {
                    transform.rotation = base.rotation * Quat::from_rotation_z(-offsets.rz);
                } else {
                    transform.rotation = base.rotation;
                }

                if offsets.has_opacity
                    && let Some(mut mat) = material
                {
                    mat.opacity = offsets.opacity;
                }
            }

            if let Some(deform_data) = deform_accum.get(&uuid.0) {
                if let Some(mut deform) = deform {
                    if deform.offsets.len() == deform_data.offsets.len() {
                        deform.offsets.copy_from_slice(&deform_data.offsets);
                    } else if !deform_data.offsets.is_empty() {
                        deform.offsets = deform_data.offsets.clone();
                    }
                }
            } else if let Some(mut deform) = deform {
                for o in deform.offsets.iter_mut() {
                    *o = [0.0, 0.0];
                }
            }
        }
    }
}

fn interpolate_binding(param: &InxParam, value: &[f32; 2], binding: &InxBinding) -> f32 {
    let InxBindingValues::Transform(flat) = &binding.values else {
        return 0.0;
    };

    let default_val = 0.0;

    if flat.frames == 0 || flat.values_per_frame == 0 || flat.data.is_empty() {
        return default_val;
    }

    let range_x = param.max[0] - param.min[0];
    let range_y = param.max[1] - param.min[1];

    let nx = if range_x.abs() > f32::EPSILON {
        ((value[0] - param.min[0]) / range_x).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let ny = if range_y.abs() > f32::EPSILON {
        ((value[1] - param.min[1]) / range_y).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let ax = &param.axis_points[0];
    let ay = &param.axis_points[1];

    if param.is_vec2 && ay.len() > 1 {
        // 2D bilinear
        let (ix, fx) = find_grid_pos(ax, nx);
        let (iy, fy) = find_grid_pos(ay, ny);

        let max_x = flat.frames.saturating_sub(1);
        let max_y = flat.values_per_frame.saturating_sub(1);

        let get = |x: usize, y: usize| -> f32 {
            let x = x.min(max_x);
            let y = y.min(max_y);
            flat.get(x, y).unwrap_or(default_val)
        };

        let v00 = get(ix, iy);
        let v10 = get(ix + 1, iy);
        let v01 = get(ix, iy + 1);
        let v11 = get(ix + 1, iy + 1);

        let top = v00 + (v10 - v00) * fx;
        let bot = v01 + (v11 - v01) * fx;
        top + (bot - top) * fy
    } else {
        // 1D linear
        let (ix, fx) = find_grid_pos(ax, nx);

        let max_x = flat.frames.saturating_sub(1);

        let get = |x: usize| -> f32 {
            let x = x.min(max_x);
            flat.get(x, 0).unwrap_or(default_val)
        };

        let v0 = get(ix);
        let v1 = get(ix + 1);
        v0 + (v1 - v0) * fx
    }
}

/// Interpola una grilla de deformación dado el param value.
/// Retorna Vec<[f32; 2]> con offsets por vértice.
fn interpolate_binding_deform(
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

    let range_x = param.max[0] - param.min[0];
    let range_y = param.max[1] - param.min[1];
    let norm_x = if range_x.abs() > f32::EPSILON {
        ((value[0] - param.min[0]) / range_x).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let norm_y = if range_y.abs() > f32::EPSILON {
        ((value[1] - param.min[1]) / range_y).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let ax = &param.axis_points[0];
    let ay = &param.axis_points[1];
    let vpf = flat.vertices_per_frame;
    let y_frames = ay.len();

    if param.is_vec2 && y_frames > 1 {
        let real_vpf = vpf / y_frames;
        let mut result = vec![[0.0f32; 2]; real_vpf];

        let (ix, fx) = find_grid_pos(ax, norm_x);
        let (iy, fy) = find_grid_pos(ay, norm_y);

        let get_vertex = |xi: usize, yi: usize, vi: usize| -> [f32; 2] {
            let xi = xi.min(ax.len().saturating_sub(1));
            let yi = yi.min(y_frames.saturating_sub(1));
            let idx = xi * vpf + yi * real_vpf + vi;
            flat.data.get(idx).copied().unwrap_or([0.0, 0.0])
        };

        for vi in 0..real_vpf {
            let v00 = get_vertex(ix, iy, vi);
            let v10 = get_vertex(ix + 1, iy, vi);
            let v01 = get_vertex(ix, iy + 1, vi);
            let v11 = get_vertex(ix + 1, iy + 1, vi);

            let top_x = v00[0] + (v10[0] - v00[0]) * fx;
            let top_y = v00[1] + (v10[1] - v00[1]) * fx;
            let bot_x = v01[0] + (v11[0] - v01[0]) * fx;
            let bot_y = v01[1] + (v11[1] - v01[1]) * fx;

            result[vi] = [
                top_x + (bot_x - top_x) * fy,
                -(top_y + (bot_y - top_y) * fy),
            ];
        }

        Some(result)
    } else {
        let mut result = vec![[0.0f32; 2]; vpf];
        let (ix, fx) = find_grid_pos(ax, norm_x);

        let get_vertex = |xi: usize, yi: usize, vi: usize| -> [f32; 2] {
            let xi = xi.min(ax.len().saturating_sub(1));
            let yi = yi.min(y_frames.saturating_sub(1));
            let idx = (xi * y_frames + yi) * vpf + vi;
            flat.data.get(idx).copied().unwrap_or([0.0, 0.0])
        };

        for vi in 0..vpf {
            let v0 = get_vertex(ix, 0, vi);
            let v1 = get_vertex(ix + 1, 0, vi);

            result[vi] = [
                v0[0] + (v1[0] - v0[0]) * fx,
                -(v0[1] + (v1[1] - v0[1]) * fx),
            ];
        }

        Some(result)
    }
}

/// Encuentra posición en la grilla de axis_points.
/// Retorna (índice_izquierdo, fracción_dentro_del_segmento).
fn find_grid_pos(axis_points: &[f32], normalized: f32) -> (usize, f32) {
    if axis_points.len() <= 1 {
        return (0, 0.0);
    }

    // axis_points está en [0, 1] (ya normalizado)
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
