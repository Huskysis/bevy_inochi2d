//! Spawns a puppet's node tree as ECS entities from a loaded [`InxPuppet`].

use bevy::{camera::visibility::RenderLayers, platform::collections::HashMap, prelude::*};

use crate::prelude::*;

/// System that processes InxScene commands and spawns the node tree.
pub fn spawn_scene_system(
    mut commands: Commands,
    query: Query<(Entity, &InxScene, Option<&RenderLayers>)>,
    puppets: Res<Assets<InxPuppet>>,
    param_assets: Res<Assets<InxParam>>,
) {
    for (entity, scene, layers) in query.iter() {
        let Some(puppet) = puppets.get(&scene.puppet) else {
            continue; // Asset not loaded yet
        };

        // Remove the command so it isn't re-processed
        commands.entity(entity).remove::<InxScene>();

        // Find root node (last node = root of the tree)
        let Some(root_node) = puppet.nodes.last() else {
            bevy::log::warn!("InxScene: puppet has no nodes");
            commands.entity(entity).despawn();
            continue;
        };

        // Spawn the tree recursively. `layers` is propagated to EVERY node (not just the root wrapper):
        // `RenderLayers` doesn't inherit through hierarchy in Bevy, so without this
        // the parts (the only entities with Mesh2d) stayed on layer 0 by default and
        // were visible to ANY camera, not just the one for the requested layer -
        // breaking the isolation `examples/rtt.rs` needs (the puppet showed up fully in the main camera, not only through the texture).
        // `uuid_map`/`deform_nodes` only live during the spawn: the uuid is resolved
        // to an Entity here and is never looked up again at runtime.
        let mut uuid_map: HashMap<u32, Entity> = HashMap::default();
        let mut deform_nodes: Vec<Entity> = Vec::new();
        let root_entity = spawn_node_recursive(
            &mut commands,
            root_node,
            scene.transform,
            None,
            layers,
            &mut uuid_map,
            &mut deform_nodes,
        );

        // Scene transform lives on a wrapper parent, NOT on the puppet's root node:
        // `evaluate_params` rewrites any bound node's Transform as base_pose +
        // offset each frame, so a scene offset placed on the root node is silently
        // destroyed when the model binds params to its root Root entity named after
        // the puppet (meta.name) so multi-puppet scenes are readable in inspectors;
        // fallback for unnamed models.
        let root_name = if puppet.meta.name.is_empty() {
            "InxScene Root".to_string()
        } else {
            puppet.meta.name.clone()
        };
        let wrapper = commands
            .spawn((
                InxPuppetRoot {
                    source: scene.puppet.clone(),
                },
                scene.transform,
                Visibility::Inherited,
                Name::new(root_name),
            ))
            .id();
        commands.entity(wrapper).add_child(root_entity);
        let root_entity = wrapper;

        // Propagate RenderLayers from the command entity to the puppet root (the command entity is despawned below).
        if let Some(layers) = layers {
            commands.entity(root_entity).insert(layers.clone());
        }

        // Param state + bindings resolved to entities
        let param_state = init_param_state(puppet, &param_assets);
        let mut resolved = InxResolvedBindings {
            deform_nodes,
            ..Default::default()
        };
        let mut mg_deform_targets: Vec<u32> = Vec::new();
        for handle in &puppet.params {
            let Some(param) = param_assets.get(handle) else {
                continue;
            };
            let bindings = param
                .bindings
                .iter()
                .enumerate()
                .filter_map(|(i, b)| uuid_map.get(&b.node_uuid).map(|&e| (i as u32, e)))
                .collect();
            for b in &param.bindings {
                if matches!(b.values, InxBindingValues::Deform(_))
                    && !mg_deform_targets.contains(&b.node_uuid)
                {
                    mg_deform_targets.push(b.node_uuid);
                }
            }
            resolved.params.push((handle.clone(), bindings));
        }
        commands.entity(root_entity).insert((param_state, resolved));

        // MeshGroup deform: build the barycentric lattice->children mapping for
        // groups some param deforms (rest poses, once per spawn).
        build_meshgroup_warps(&mut commands, root_node, &uuid_map, &mg_deform_targets);

        if scene.default_pose {
            commands.entity(root_entity).insert(InxDefaultPose);
        }

        // Animation (if applicable)
        if scene.animation {
            let mut controller = InxAnimationController::new();

            // Param defaults
            for param_handle in &puppet.params {
                if let Some(param) = param_assets.get(param_handle) {
                    controller.param_defaults.insert(param.uuid, param.defaults);
                }
            }

            // All animations are registered but NOT played automatically. The user
            // decides what to play afterwards via controller.play() / set_idle().
            commands.entity(root_entity).insert(controller);
        }

        // Remove the temporary command entity
        commands.entity(entity).despawn();
    }
}

/// Spawns a node and its children recursively. Returns the root entity.
///
/// `composite_ancestor` carries the innermost enclosing composite-group entity (if any).
/// Each non-root entity inherits it as `InComposite(...)` so the Mesh2d renderer can
/// route draws to the right composite pass.
fn spawn_node_recursive(
    commands: &mut Commands,
    node: &InxNode,
    _parent_transform: Transform,
    composite_ancestor: Option<Entity>,
    layers: Option<&RenderLayers>,
    uuid_map: &mut HashMap<u32, Entity>,
    deform_nodes: &mut Vec<Entity>,
) -> Entity {
    let final_transform = Transform {
        translation: Vec3::new(
            node.transform.translation.x,
            -node.transform.translation.y,
            node.transform.translation.z,
        ),
        rotation: Quat::from_rotation_z(-node.transform.rotation.z),
        scale: Vec3::new(node.transform.scale.x, node.transform.scale.y, 1.0),
    };

    let entity = if let Some(mat) = &node.material {
        let mut ec = commands.spawn((
            InxUUID(node.uuid),
            InxZSort(node.zsort),
            node.node_type,
            mat.clone(),
            final_transform,
            if node.enabled {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            },
            Name::new(node.name.to_string()),
            InxBasePose {
                translation: final_transform.translation,
                rotation: final_transform.rotation,
                scale: final_transform.scale,
            },
        ));
        if let Some(mesh) = &mat.mesh {
            ec.insert(InxDeform {
                offsets: vec![[0.0, 0.0]; mesh.vertex_buffer.len()],
            });
            deform_nodes.push(ec.id());
        }
        ec.id()
    } else {
        commands
            .spawn((
                InxUUID(node.uuid),
                InxZSort(node.zsort),
                node.node_type,
                final_transform,
                if node.enabled {
                    Visibility::Inherited
                } else {
                    Visibility::Hidden
                },
                Name::new(node.name.to_string()),
                InxBasePose {
                    translation: final_transform.translation,
                    rotation: final_transform.rotation,
                    scale: final_transform.scale,
                },
            ))
            .id()
    };

    // SimplePhysics
    if node.node_type == InxNodeType::SimplePhysics
        && let Some(phys) = &node.physics_data
        && phys.param_uuid != u32::MAX
    {
        commands.entity(entity).insert((
            InxSimplePhysics {
                param_uuid: phys.param_uuid,
                model: phys.model,
                map_mode: phys.map_mode,
                gravity: phys.gravity,
                length: phys.length,
                frequency: phys.frequency,
                angle_damping: phys.angle_damping,
                length_damping: phys.length_damping,
                output_scale: phys.output_scale,
                local_only: phys.local_only,
            },
            InxPhysicsState::default(),
            GlobalTransform::default(),
        ));
    }

    // Tag descendants of a composite so the Mesh2d renderer can route their draws to
    // the composite pass.
    if let Some(ancestor) = composite_ancestor {
        commands.entity(entity).insert(InComposite(ancestor));
    }

    // RenderLayers doesn't propagate through Bevy hierarchy - every node needs its
    // own copy for camera visibility filtering to work.
    if let Some(layers) = layers {
        commands.entity(entity).insert(layers.clone());
    }

    // Composite group: lift composite fields from `InxMaterial` to a dedicated
    // component. Empty composite (no children) is skipped with a warning. Nested
    // composites: the inner group's own entity keeps its
    // `InComposite(composite_ancestor)` from the tagging above, and its final quad
    // (attach_composite_quads) inherits that same tag, so the outer group treats it
    // as a regular member for bbox/Z/render order.
    let mut child_ancestor = composite_ancestor;
    if node.node_type == InxNodeType::Composite {
        if node.children.is_empty() {
            bevy::log::warn!(
                "composite '{}' has no children - skipping group",
                node.name
            );
        } else {
            let mat = node.material.as_ref();
            let group = InxCompositeGroup {
                blend_mode: mat.map(|m| m.blend_mode).unwrap_or(BlendMode::Normal),
                opacity: mat.map(|m| m.opacity).unwrap_or(1.0),
                tint: mat.map(|m| m.tint).unwrap_or(Vec3::ONE),
                screen_tint: mat.map(|m| m.screen_tint).unwrap_or(Vec3::ZERO),
                mask_threshold: mat.map(|m| m.mask_threshold).unwrap_or(0.5),
                padding: 8.0,
                zsort: node.zsort,
            };
            let mode = composite::classify(&group, node.compose_hint);
            commands
                .entity(entity)
                .insert((group, mode, InxCompositeBbox::default()));
            child_ancestor = Some(entity);
        }
    }

    uuid_map.insert(node.uuid, entity);

    // Children - direct recursion over Vec<InxNode>
    for child_node in &node.children {
        let child_entity = spawn_node_recursive(
            commands,
            child_node,
            Transform::IDENTITY,
            child_ancestor,
            layers,
            uuid_map,
            deform_nodes,
        );
        commands.entity(entity).add_child(child_entity);
    }

    entity
}

/// 2D affine in raw INR space (y-down, pre Bevy sign flips): p' = l*p + t.
#[derive(Clone, Copy)]
struct Aff2 {
    l: [[f32; 2]; 2],
    t: [f32; 2],
}

impl Aff2 {
    const IDENTITY: Aff2 = Aff2 {
        l: [[1.0, 0.0], [0.0, 1.0]],
        t: [0.0, 0.0],
    };

    /// Local transform of a node in RENDER space - the same convention the spawner
    /// puts on entities (INR translation y negated, rotation z negated), because
    /// mesh vertex buffers and deform offsets are consumed raw in that space
    /// (`part_positions`). Mixing raw INR translations with render-space vertices
    /// mirrors the mapping vertically.
    fn from_node(node: &InxNode) -> Aff2 {
        let (_, _, rz) = node.transform.rotation.to_euler(EulerRot::XYZ);
        let (s, c) = (-rz).sin_cos();
        let (sx, sy) = (node.transform.scale.x, node.transform.scale.y);
        Aff2 {
            l: [[c * sx, -s * sy], [s * sx, c * sy]],
            t: [node.transform.translation.x, -node.transform.translation.y],
        }
    }

    /// self ∘ other: apply `other` first, then `self`.
    fn compose(&self, other: &Aff2) -> Aff2 {
        let a = &self.l;
        let b = &other.l;
        Aff2 {
            l: [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1],
                ],
            ],
            t: [
                a[0][0] * other.t[0] + a[0][1] * other.t[1] + self.t[0],
                a[1][0] * other.t[0] + a[1][1] * other.t[1] + self.t[1],
            ],
        }
    }

    fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        [
            self.l[0][0] * p[0] + self.l[0][1] * p[1] + self.t[0],
            self.l[1][0] * p[0] + self.l[1][1] * p[1] + self.t[1],
        ]
    }

    fn inv_linear(&self) -> [[f32; 2]; 2] {
        let det = self.l[0][0] * self.l[1][1] - self.l[0][1] * self.l[1][0];
        if det.abs() < 1e-9 {
            return [[1.0, 0.0], [0.0, 1.0]];
        }
        let inv = 1.0 / det;
        [
            [self.l[1][1] * inv, -self.l[0][1] * inv],
            [-self.l[1][0] * inv, self.l[0][0] * inv],
        ]
    }
}

/// Barycentric coords of `p` in triangle (a, b, c); None if degenerate.
pub(crate) fn barycentric(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Option<[f32; 3]> {
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];
    let den = v0[0] * v1[1] - v1[0] * v0[1];
    if den.abs() < 1e-9 {
        return None;
    }
    let v = (v2[0] * v1[1] - v1[0] * v2[1]) / den;
    let w = (v0[0] * v2[1] - v2[0] * v0[1]) / den;
    Some([1.0 - v - w, v, w])
}

/// Maps one part's vertices into a MeshGroup lattice: containing triangle per
/// vertex, or `None` if outside every lattice triangle at rest.
///
/// Matches upstream MeshGroup.filterChildren (Inochi2D v0.8.7, meshgroup/package.d):
/// a vertex outside the rasterized triangle bitmask gets `index = -1` -> `newPos =
/// cVertex` -> zero warp, left untouched. Extrapolating from the nearest triangle
/// instead produces wild offsets at parameter extremes, so vertices off the lattice are left unwarped.
#[allow(clippy::type_complexity)]
fn map_part_to_lattice(
    lattice: &InxMesh,
    part_mesh: &InxMesh,
    part_to_group: &Aff2,
) -> (Vec<Option<([u32; 3], [f32; 3])>>, Vec<[f32; 2]>) {
    let lat_origin = lattice.origin;
    let tris: Vec<[u32; 3]> = lattice
        .index_buffer
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let lat_pos = |i: u32| {
        let v = lattice.vertex_buffer[i as usize];
        [v[0] - lat_origin.x, v[1] - lat_origin.y]
    };
    part_mesh
        .vertex_buffer
        .iter()
        .map(|v| {
            let local = [v[0] - part_mesh.origin.x, v[1] - part_mesh.origin.y];
            let p = part_to_group.apply(local);
            let m: Option<([u32; 3], [f32; 3])> = tris.iter().find_map(|tri| {
                let bary = barycentric(p, lat_pos(tri[0]), lat_pos(tri[1]), lat_pos(tri[2]))?;
                (bary[0].min(bary[1]).min(bary[2]) >= 0.0).then_some((*tri, bary))
            });
            (m, p)
        })
        .unzip()
}

/// Lattice rest positions (origin-adjusted, group-local) + triangle indices.
fn lattice_rest_and_tris(lattice: &InxMesh) -> (Vec<[f32; 2]>, Vec<[u32; 3]>) {
    let o = lattice.origin;
    let rest = lattice
        .vertex_buffer
        .iter()
        .map(|v| [v[0] - o.x, v[1] - o.y])
        .collect();
    let tris = lattice
        .index_buffer
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    (rest, tris)
}

/// Builds `InxMeshGroupWarp` for every MeshGroup in `mg_deform_targets` (groups some param binds with target=Deform):
/// barycentric mapping of every descendant part's vertices through the group
/// lattice, rest poses, INR space.
fn build_meshgroup_warps(
    commands: &mut Commands,
    root: &InxNode,
    uuid_map: &HashMap<u32, Entity>,
    mg_deform_targets: &[u32],
) {
    // (group uuid, lattice, affine current-node-local -> group space)
    struct ActiveGroup<'a> {
        uuid: u32,
        lattice: &'a InxMesh,
        to_group: Aff2,
        children: Vec<MgChildMap>,
        dynamic: bool,
    }

    // A finished group ready to become an `InxMeshGroupWarp`.
    struct BuiltGroup {
        uuid: u32,
        children: Vec<MgChildMap>,
        lattice_rest: Vec<[f32; 2]>,
        tris: Vec<[u32; 3]>,
        dynamic: bool,
    }

    fn walk<'a>(
        node: &'a InxNode,
        active: &mut Vec<ActiveGroup<'a>>,
        // Number of leading `active` groups suspended in this subtree: a Composite
        // with propagate_meshgroup=false is a barrier, so ancestor MeshGroups
        // (opened before it) must not warp its descendants - only groups opened at
        // or after the barrier (index >= suspend_before) do.
        suspend_before: usize,
        uuid_map: &HashMap<u32, Entity>,
        mg_deform_targets: &[u32],
        out: &mut Vec<BuiltGroup>,
    ) {
        let local = Aff2::from_node(node);
        // Advance every already-active group's affine through this node.
        let saved: Vec<Aff2> = active.iter().map(|g| g.to_group).collect();
        for g in active.iter_mut() {
            g.to_group = g.to_group.compose(&local);
        }

        let mut opened = false;
        if node.node_type == InxNodeType::MeshGroup
            && mg_deform_targets.contains(&node.uuid)
            && let Some(lattice) = node.mesh.as_deref()
        {
            active.push(ActiveGroup {
                uuid: node.uuid,
                lattice,
                to_group: Aff2::IDENTITY,
                children: Vec::new(),
                dynamic: node.mesh_group_dynamic,
            });
            opened = true;
        }

        if node.node_type == InxNodeType::Part
            && let Some(mesh) = node.material.as_ref().and_then(|m| m.mesh.as_deref())
            && let Some(&entity) = uuid_map.get(&node.uuid)
        {
            for g in active[suspend_before..].iter_mut() {
                let (map, rest_query) = map_part_to_lattice(g.lattice, mesh, &g.to_group);
                if map.iter().any(|m| m.is_some()) {
                    g.children.push(MgChildMap {
                        entity,
                        map,
                        inv_linear: g.to_group.inv_linear(),
                        fwd_linear: g.to_group.l,
                        rest_query,
                    });
                }
            }
        }

        // A non-propagating composite suspends every currently-active ancestor group
        // for its subtree; groups opened inside it still apply.
        let child_suspend = if node.node_type == InxNodeType::Composite
            && !node.composite_propagate_meshgroup
        {
            active.len()
        } else {
            suspend_before
        };
        for child in &node.children {
            walk(child, active, child_suspend, uuid_map, mg_deform_targets, out);
        }

        if opened {
            let g = active.pop().expect("balanced push/pop");
            if !g.children.is_empty() {
                let (lattice_rest, tris) = lattice_rest_and_tris(g.lattice);
                out.push(BuiltGroup {
                    uuid: g.uuid,
                    children: g.children,
                    lattice_rest,
                    tris,
                    dynamic: g.dynamic,
                });
            }
        }
        // Restore the affines this node advanced.
        for (g, s) in active.iter_mut().zip(saved) {
            g.to_group = s;
        }
    }

    let mut out: Vec<BuiltGroup> = Vec::new();
    let mut active: Vec<ActiveGroup> = Vec::new();
    walk(root, &mut active, 0, uuid_map, mg_deform_targets, &mut out);
    for g in out {
        if let Some(&entity) = uuid_map.get(&g.uuid) {
            commands.entity(entity).insert(InxMeshGroupWarp {
                children: g.children,
                lattice_rest: g.lattice_rest,
                tris: g.tris,
                dynamic: g.dynamic,
            });
        }
    }
}

/// Initializes InxParamState with the puppet's defaults.
fn init_param_state(puppet: &InxPuppet, params: &Assets<InxParam>) -> InxParamState {
    let mut state = InxParamState::default();
    for handle in &puppet.params {
        if let Some(param) = params.get(handle) {
            state.values.insert(param.uuid, param.defaults);
        }
    }
    state
}
