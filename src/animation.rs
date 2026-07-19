//! Multi-layer animation controller, default-pose application and per-frame param evaluation.

use bevy::{platform::collections::HashMap, prelude::*};

use crate::prelude::*;

/// Advances time for each layer, updates fades, evaluates lanes, and blends the
/// weighted results into InxParamState.
pub fn update_animation_controller(
    time: Res<Time>,
    animations: Res<Assets<InxAnimation>>,
    mut query: Query<(&mut InxAnimationController, &mut InxParamState)>,
) {
    let dt = time.delta_secs();

    for (mut controller, mut state) in query.iter_mut() {
        // Paused: layer time/fade doesn't advance and no layer is removed - the
        // frame stays frozen as-is, but the evaluation below still runs with
        // `layer.time` frozen, so the params touched do NOT fall into the
        // reset-to-default further below (unlike `stop_all`, which does reset).
        if !controller.paused {
            // tick layers (time + fade)
            for layer in controller.layers.iter_mut() {
                if !layer.playing {
                    continue;
                }

                // Advance time
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

                // Update fade
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

            // Clean up dead layers (weight 0 + not playing), except layer 0 (idle)
            let layer_count = controller.layers.len();
            if layer_count > 1 {
                controller
                    .layers
                    .retain_mut(|l| l.playing || l.weight > 0.001);
                // Ensure at least one always remains
                if controller.layers.is_empty() {
                    // Shouldn't happen, but just in case
                }
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
                // Lanes within the same animation apply normal merge_mode
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

        // Do NOT replace state.values entirely (would erase physics). Only overwrite
        // params that animations touch.

        // Collect all param UUIDs touched by any layer
        let mut touched: HashMap<u32, [f32; 2]> = HashMap::default();

        for (weight, values) in &layer_values {
            for (&uuid, &layer_val) in values {
                let base = controller
                    .param_defaults
                    .get(&uuid)
                    .copied()
                    .unwrap_or([0.0, 0.0]);

                let current = touched.entry(uuid).or_insert(base);
                current[0] = lerp(current[0], layer_val[0], *weight);
                current[1] = lerp(current[1], layer_val[1], *weight);
            }
        }

        // Merge: overwrite touched params; those NOT touched by any active layer
        // revert to their default (otherwise they stay frozen at the last value after stop_all()/fade-out).
        // Safe for physics-driven params: the physics system runs AFTER this in the
        // schedule and always overwrites its own param_uuid every frame.
        for (&uuid, &default) in &controller.param_defaults {
            let val = touched.get(&uuid).copied().unwrap_or(default);
            // Compare-before-write: an unconditional insert marks InxParamState
            // Changed every frame and forces evaluate_params (and everything downstream)
            // to rerun on puppets at rest.
            if state.values.get(&uuid) != Some(&val) {
                state.values.insert(uuid, val);
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

type NodeQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static InxBasePose,
        &'static mut Transform,
        Option<&'static mut InxMaterial>,
        Option<&'static mut InxDeform>,
    ),
>;

/// Resolves every param binding for one puppet and writes the result (`base_pose + offset`)
/// into its nodes' `Transform`/`InxMaterial`/ `InxDeform`. A param with no entry in
/// `state.values` falls back to its authored default - this is what makes a fresh
/// `InxParamState` (no animation ever played) already produce the puppet's rest
/// pose, not raw authored transforms.
///
/// Bindings arrive already resolved to `Entity` ([`InxResolvedBindings`], built at spawn):
/// there are no uuid lookups or world scans - each puppet writes exactly to its own nodes.
fn apply_params_for_puppet(
    state: &InxParamState,
    resolved: &InxResolvedBindings,
    param_assets: &Assets<InxParam>,
    node_query: &mut NodeQuery,
    warps: &Query<&crate::InxMeshGroupWarp>,
) {
    // accumulators (several params can bind the same node)
    let mut transform_accum: HashMap<Entity, NodeAccum> = HashMap::default();
    let mut deform_accum: HashMap<Entity, DeformAccum> = HashMap::default();

    for (param_handle, bindings) in &resolved.params {
        let Some(param) = param_assets.get(param_handle) else {
            continue;
        };

        let param_value = state
            .values
            .get(&param.uuid)
            .copied()
            .unwrap_or(param.defaults);

        for &(binding_idx, target) in bindings {
            let Some(binding) = param.bindings.get(binding_idx as usize) else {
                continue;
            };
            match &binding.values {
                InxBindingValues::Transform(_) => {
                    let value = interpolate_binding(param, &param_value, binding);
                    let entry = transform_accum.entry(target).or_default();

                    entry.accumulate(&binding.param_name, value, param.merge_mode);
                }
                InxBindingValues::Deform(_) => {
                    if let Some(offsets) = interpolate_binding_deform(param, &param_value, binding)
                    {
                        let entry = deform_accum.entry(target).or_insert_with(|| DeformAccum {
                            offsets: vec![[0.0; 2]; offsets.len()],
                            _initialized: false,
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

    // MeshGroup deform: a group's accumulated lattice offsets warp its descendant
    // parts' vertices (static barycentric mapping built at spawn). Distribute before
    // the write-back so children compose group warp with their own deform bindings
    // additively. Nested groups each map all their descendant parts directly, so
    // contributions stack here too.
    let group_entities: Vec<Entity> = deform_accum
        .keys()
        .copied()
        .filter(|e| warps.contains(*e))
        .collect();
    // MeshGroup deform, per group's `dynamic_deformation` flag:
    // - dynamic: re-locate each child vertex by its own deform first (query = rest + own_deform, group space),
    //   recompute the containing lattice triangle against the deformed lattice, take
    //   the full displacement- `group_warp(rest+own)`. Matches upstream's runtime
    //   warp for dynamic MeshGroups.
    // - static: warp the rest-pose barycentric coords, additive - `own +
    //   group_warp(rest)`. Upstream bakes these at export, but we apply the
    //   equivalent at runtime.
    for group in group_entities {
        let Some(group_offsets) = deform_accum.remove(&group) else {
            continue;
        };
        let Ok(warp) = warps.get(group) else { continue };
        let dynamic = warp.dynamic;
        for child in &warp.children {
            let entry = deform_accum
                .entry(child.entity)
                .or_insert_with(|| DeformAccum {
                    offsets: vec![[0.0; 2]; child.map.len()],
                    _initialized: false,
                });
            if entry.offsets.len() != child.map.len() {
                continue;
            }
            let il = &child.inv_linear;
            if dynamic {
                let fl = &child.fwd_linear;
                // Group lattice deformed positions (rest + group offset).
                let def_lat: Vec<[f32; 2]> = warp
                    .lattice_rest
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let o = group_offsets.offsets.get(i).copied().unwrap_or([0.0, 0.0]);
                        [r[0] + o[0], r[1] + o[1]]
                    })
                    .collect();
                for (vi, acc) in entry.offsets.iter_mut().enumerate() {
                    let rq = child.rest_query[vi];
                    // Shift query by the child's own deform, mapped to group space.
                    let od = *acc; // own deform accumulated so far (child-local)
                    let q = [
                        rq[0] + fl[0][0] * od[0] + fl[0][1] * od[1],
                        rq[1] + fl[1][0] * od[0] + fl[1][1] * od[1],
                    ];
                    // Locate q in the REST lattice to get barycentric coords.
                    let found = warp.tris.iter().find_map(|tri| {
                        let a = warp.lattice_rest[tri[0] as usize];
                        let b = warp.lattice_rest[tri[1] as usize];
                        let c = warp.lattice_rest[tri[2] as usize];
                        let bary = crate::auto_spawn::barycentric(q, a, b, c)?;
                        (bary[0].min(bary[1]).min(bary[2]) >= 0.0).then_some((*tri, bary))
                    });
                    let Some((tri, bary)) = found else { continue };
                    // Deformed position via the same barys on the deformed lattice.
                    let mut warped = [0.0f32, 0.0];
                    for (t, bw) in tri.iter().zip(bary.iter()) {
                        let d = def_lat[*t as usize];
                        warped[0] += d[0] * bw;
                        warped[1] += d[1] * bw;
                    }
                    // Group-space displacement of the query point, back to
                    // child-local.
                    let gx = warped[0] - q[0];
                    let gy = warped[1] - q[1];
                    acc[0] += il[0][0] * gx + il[0][1] * gy;
                    acc[1] += il[1][0] * gx + il[1][1] * gy;
                }
            } else {
                for (acc, m) in entry.offsets.iter_mut().zip(child.map.iter()) {
                    let Some((tri, bary)) = m else { continue };
                    let mut dx = 0.0;
                    let mut dy = 0.0;
                    for (t, b) in tri.iter().zip(bary.iter()) {
                        if let Some(o) = group_offsets.offsets.get(*t as usize) {
                            dx += o[0] * b;
                            dy += o[1] * b;
                        }
                    }
                    acc[0] += il[0][0] * dx + il[0][1] * dy;
                    acc[1] += il[1][0] * dx + il[1][1] * dy;
                }
            }
        }
    }

    // Compare-before-write everywhere below: an unconditional write marks the
    // component Changed even with identical values, which defeats Bevy's change
    // detection downstream (mask clipping, deform sync, transform propagation would re-run every frame on puppets at rest).
    for (&entity, offsets) in &transform_accum {
        let Ok((base, mut transform, material, _)) = node_query.get_mut(entity) else {
            continue;
        };

        let translation = Vec3::new(
            base.translation.x + offsets.tx,
            base.translation.y - offsets.ty,
            base.translation.z + offsets.tz,
        );
        let scale = Vec3::new(base.scale.x * offsets.sx, base.scale.y * offsets.sy, 1.0);
        let rotation = if offsets.rz.abs() > 1e-6 {
            base.rotation * Quat::from_rotation_z(-offsets.rz)
        } else {
            base.rotation
        };

        if transform.translation != translation
            || transform.scale != scale
            || transform.rotation != rotation
        {
            transform.translation = translation;
            transform.scale = scale;
            transform.rotation = rotation;
        }

        if offsets.has_opacity
            && let Some(mut mat) = material
            && mat.opacity != offsets.opacity
        {
            mat.opacity = offsets.opacity;
        }
    }

    // Deforms: bound node receives the accumulated value; the rest go back to zero
    // (same contract as before, but only over THIS puppet's nodes).
    for &entity in &resolved.deform_nodes {
        let Ok((_, _, _, Some(mut deform))) = node_query.get_mut(entity) else {
            continue;
        };
        if let Some(deform_data) = deform_accum.get(&entity) {
            if deform.offsets == deform_data.offsets {
                continue;
            }
            if deform.offsets.len() == deform_data.offsets.len() {
                deform.offsets.copy_from_slice(&deform_data.offsets);
            } else if !deform_data.offsets.is_empty() {
                deform.offsets = deform_data.offsets.clone();
            }
        } else if deform.offsets.iter().any(|o| *o != [0.0, 0.0]) {
            for o in deform.offsets.iter_mut() {
                *o = [0.0, 0.0];
            }
        }
    }
}

/// Resolves every puppet's `InxParamState` (written by `update_animation_controller`)
/// into its nodes' `Transform`/`InxMaterial`/`InxDeform`. Part of
/// `InxAnimationPlugin`'s per-frame loop - see [`apply_default_pose`] for the same
/// resolution without the animation loop.
pub fn evaluate_params(
    puppet_query: Query<(Ref<InxParamState>, &InxResolvedBindings)>,
    param_assets: Res<Assets<InxParam>>,
    mut node_query: NodeQuery,
    warps: Query<&crate::InxMeshGroupWarp>,
) {
    for (state, resolved) in puppet_query.iter() {
        // Skip puppets whose params didn't move since the last run (all writers compare before writing)
        // - resolving every binding is the per-frame hot loop on big models.
        if !state.is_changed() {
            continue;
        }
        apply_params_for_puppet(&state, resolved, &param_assets, &mut node_query, &warps);
    }
}

/// Same resolution as [`evaluate_params`], scoped to puppets tagged
/// [`crate::InxDefaultPose`]- lets a scene get its rest pose (params at authored defaults)
/// without registering `InxAnimationPlugin`'s full per-frame loop
/// (`update_animation_controller` + `simple_physics_system`). Don't combine both on
/// the same puppet: with `InxAnimationPlugin` also driving it, the two systems race
/// to write the same `Transform` each frame.
pub fn apply_default_pose(
    puppet_query: Query<(&InxParamState, &InxResolvedBindings), With<crate::InxDefaultPose>>,
    param_assets: Res<Assets<InxParam>>,
    mut node_query: NodeQuery,
    warps: Query<&crate::InxMeshGroupWarp>,
) {
    for (state, resolved) in puppet_query.iter() {
        apply_params_for_puppet(state, resolved, &param_assets, &mut node_query, &warps);
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

/// Interpolates a deformation grid given the param value. Returns Vec<[f32; 2]> with offsets per vertex.
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

        for (vi, slot) in result.iter_mut().enumerate() {
            let v00 = get_vertex(ix, iy, vi);
            let v10 = get_vertex(ix + 1, iy, vi);
            let v01 = get_vertex(ix, iy + 1, vi);
            let v11 = get_vertex(ix + 1, iy + 1, vi);

            let top_x = v00[0] + (v10[0] - v00[0]) * fx;
            let top_y = v00[1] + (v10[1] - v00[1]) * fx;
            let bot_x = v01[0] + (v11[0] - v01[0]) * fx;
            let bot_y = v01[1] + (v11[1] - v01[1]) * fx;

            *slot = [
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

        for (vi, slot) in result.iter_mut().enumerate() {
            let v0 = get_vertex(ix, 0, vi);
            let v1 = get_vertex(ix + 1, 0, vi);

            *slot = [
                v0[0] + (v1[0] - v0[0]) * fx,
                -(v0[1] + (v1[1] - v0[1]) * fx),
            ];
        }

        Some(result)
    }
}

/// Finds position in the axis_points grid. Returns (left_index, fraction_within_segment).
fn find_grid_pos(axis_points: &[f32], normalized: f32) -> (usize, f32) {
    if axis_points.len() <= 1 {
        return (0, 0.0);
    }

    // axis_points is in [0, 1] (already normalized)
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
