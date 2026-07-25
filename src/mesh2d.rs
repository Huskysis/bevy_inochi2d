//! Mesh2d/Material2d renderer - the crate's only renderer.
//!
//! Parts are regular `Mesh2d` + `MeshMaterial2d<InxPartMaterial>` entities, so Bevy
//! sprites/meshes interleave natively via Z. Covers blend-mode specialization,
//! CPU-clipped masks and composite groups (`composite.rs`), with gaps in some
//! complex blend modes, `mask_threshold` and `Camera` nodes.

use bevy::{
    asset::embedded_asset,
    mesh::{Indices, Mesh2d, PrimitiveTopology},
    prelude::*,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
    sprite_render::{AlphaMode2d, Material2d, Material2dKey, Material2dPlugin},
};

use bevy::transform::TransformSystems;

use std::collections::{HashMap, HashSet};

use i_overlay::core::{fill_rule::FillRule, overlay_rule::OverlayRule};
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::i_shape::base::data::Shapes;
use i_triangle::float::triangulatable::Triangulatable;

use crate::{
    BlendMode, InxDeform, InxMaskMode, InxMaterial, InxNodeType, InxPuppetRoot, InxUUID,
    InxZSort,
    composite::{
        ComposeMode, InComposite, InxCompositeGroup, InxCompositeQuad, acquire_composite_rts,
        composite_pass, extract_composites, queue_composite_views, update_composite_bbox,
    },
    plugin::Inochi2dCorePlugin,
};

use bevy::render::{
    ExtractSchedule, Render, RenderApp, RenderSystems,
    renderer::{RenderGraph, RenderGraphSystems},
};

/// The Mesh2d/Material2d renderer. Adds [`Inochi2dCorePlugin`] itself if not already
/// present, so `Inochi2dPlugin` (core + this) is the usual entry point.
pub struct InxMesh2dPlugin;

impl Plugin for InxMesh2dPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "part2d.wgsl");
        if !app.is_plugin_added::<Inochi2dCorePlugin>() {
            app.add_plugins(Inochi2dCorePlugin);
        }
        app.init_resource::<InxZBand>()
            .add_plugins(Material2dPlugin::<InxPartMaterial>::default())
            .add_systems(
                Update,
                (attach_part_meshes, link_part_masks).chain(),
            )
            .add_systems(Update, attach_composite_quads)
            .add_systems(
                Update,
                crate::composite::isolate_needs_rt_children,
            )
            .add_systems(
                PostUpdate,
                crate::composite::force_composite_member_visibility
                    .in_set(bevy::camera::visibility::VisibilitySystems::CheckVisibility),
            )
            .add_systems(
                PostUpdate,
                (sync_part_deforms, sync_part_materials).before(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                (sync_part_z, sync_mask_clipping)
                    .chain()
                    .after(TransformSystems::Propagate),
            )
            .add_systems(
                // Composite rendering is a core feature, not tied to animation -
                // must run whether or not InxAnimationPlugin is present. Needs
                // GlobalTransform freshly propagated (from whatever wrote Transform this frame: evaluate_params if animation is active, or the static base pose otherwise).
                PostUpdate,
                update_composite_bbox.after(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                (
                    acquire_composite_rts.after(update_composite_bbox),
                    sync_composite_quads
                        .after(acquire_composite_rts)
                        .after(sync_part_z),
                ),
            );

        // Offscreen fallback: NeedsRt composites are mirrored into the render world.
        // Headless apps without a RenderApp skip this.
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<crate::composite::ExtractedComposites>()
                .init_resource::<crate::composite::CompositeViewEntities>()
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_composites,
                        // After Bevy's phase extraction: its retain() only keeps
                        // camera views and would drop our phases. Also after the
                        // dirty-specialization clear, so marking our view dirty sticks.
                        queue_composite_views
                            .after(bevy::core_pipeline::core_2d::extract_core_2d_camera_phases)
                            .after(bevy::render::camera::DirtySpecializationSystems::Clear),
                    )
                        .chain(),
                )
                .init_resource::<crate::composite::CompositeDepthTextures>()
                .add_schedule(bevy::ecs::schedule::Schedule::new(
                    crate::composite::InxCompositeViewSchedule,
                ))
                .add_systems(
                    Render,
                    crate::composite::prepare_composite_depth_textures
                        .in_set(RenderSystems::PrepareResources),
                )
                // Once per frame in the root render schedule, before the camera
                // schedules run: composite RTs must be filled before the main 2D
                // pass draws the quads that sample them.
                .add_systems(
                    RenderGraph,
                    composite_pass
                        .in_set(RenderGraphSystems::Render)
                        .before(bevy::core_pipeline::schedule::camera_driver),
                );
        }
    }
}

/// GPU uniform for `part2d.wgsl`, mirrored from [`InxMaterial`]/[`crate::composite::InxCompositeGroup`].
#[derive(ShaderType, Clone, Debug)]
pub struct InxPartMaterialUniform {
    /// Additive RGB tint.
    pub tint: Vec3,
    /// Global opacity.
    pub opacity: f32,
    /// Screen tint.
    pub screen_tint: Vec3,
    /// 1 = composite quad: the sampled texture (a composite RT) is already
    /// premultiplied, so the shader skips its straight-alpha premultiply.
    pub composite: u32,
    /// Multiplier for the emissive texture; 0 when the part has none.
    pub emissive_strength: f32,
}

/// The `Material2d` every puppet part/prop/composite-quad renders with. `blend_mode`
/// drives pipeline specialization ([`InxPartMaterialKey`]), not just a uniform value
/// - each mode needs its own fixed-function blend state.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
#[bind_group_data(InxPartMaterialKey)]
pub struct InxPartMaterial {
    /// Tint/opacity/composite uniform.
    #[uniform(0)]
    pub data: InxPartMaterialUniform,
    /// Albedo texture.
    #[texture(1)]
    #[sampler(2)]
    pub albedo: Option<Handle<Image>>,
    /// Emissive texture.
    #[texture(3)]
    #[sampler(4)]
    pub emissive: Option<Handle<Image>>,
    /// Bound for completeness (INR slot 2); the 2D fragment shader has no lighting
    /// stage yet, so it is currently unread.
    #[texture(5)]
    #[sampler(6)]
    pub bumpmap: Option<Handle<Image>>,
    /// Blend mode, drives pipeline specialization.
    pub blend_mode: BlendMode,
}

/// Specialization key: one pipeline variant per [`BlendMode`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct InxPartMaterialKey {
    /// Blend mode this pipeline variant is specialized for.
    pub blend_mode: BlendMode,
}

impl From<&InxPartMaterial> for InxPartMaterialKey {
    fn from(material: &InxPartMaterial) -> Self {
        Self {
            blend_mode: material.blend_mode,
        }
    }
}

impl Material2d for InxPartMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_inochi2d/part2d.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }

    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The shader outputs premultiplied alpha; per-mode fixed-function blend
        // states cover all 19 Inochi2D blend modes.
        let blend = key.bind_group_data.blend_mode.blend_state();
        if let Some(fragment) = &mut descriptor.fragment {
            for target in fragment.targets.iter_mut().flatten() {
                target.blend = Some(blend);
            }
        }
        Ok(())
    }
}

fn part_positions(mat: &InxMaterial, deform: Option<&InxDeform>) -> Option<Vec<[f32; 3]>> {
    let mesh = mat.mesh.as_ref()?;
    let origin = mesh.origin;
    let mut positions = Vec::with_capacity(mesh.vertex_buffer.len());
    for (i, v) in mesh.vertex_buffer.iter().enumerate() {
        let d = deform
            .and_then(|d| d.offsets.get(i))
            .copied()
            .unwrap_or([0.0, 0.0]);
        positions.push([v[0] - origin.x + d[0], v[1] - origin.y + d[1], 0.0]);
    }
    Some(positions)
}

/// Inserts Mesh2d + material on every Part spawned by `spawn_scene_system`.
#[allow(clippy::type_complexity)] // Bevy Query filter types
pub fn attach_part_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<InxPartMaterial>>,
    parts: Query<
        (
            Entity,
            &InxMaterial,
            &InxNodeType,
            Option<&InxDeform>,
            Option<&InComposite>,
        ),
        (Without<Mesh2d>, With<InxZSort>),
    >,
    groups: Query<(&InxMaterial, &ComposeMode), With<InxCompositeGroup>>,
) {
    for (entity, mat, node_type, deform, member) in parts.iter() {
        if *node_type != InxNodeType::Part {
            continue;
        }
        let Some(inx_mesh) = mat.mesh.as_ref() else {
            continue;
        };
        let Some(positions) = part_positions(mat, deform) else {
            continue;
        };

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, inx_mesh.uv_buffer.clone());
        mesh.insert_indices(Indices::U32(inx_mesh.index_buffer.clone()));

        let group = per_child_blend_group(member, &groups);
        let (data, blend_mode) = part_material_values(mat, group);
        let material = materials.add(InxPartMaterial {
            data,
            albedo: mat.texture_albedo.clone(),
            emissive: mat.texture_emissive.clone(),
            bumpmap: mat.texture_bumpmap.clone(),
            blend_mode,
        });

        commands
            .entity(entity)
            .insert((Mesh2d(meshes.add(mesh)), MeshMaterial2d(material)));
    }
}

/// Spawn the final quad for every `NeedsRt` composite: a standalone unit-square
/// Mesh2d whose material samples the composite RT. Placement/size happen per frame
/// in [`sync_composite_quads`]; Z comes from [`sync_part_z`] (the group's atomic rank).
#[allow(clippy::type_complexity)] // Bevy Query filter types
pub fn attach_composite_quads(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<InxPartMaterial>>,
    groups: Query<(
        Entity,
        &ComposeMode,
        &InxCompositeGroup,
        Option<&InComposite>,
        Option<&bevy::camera::visibility::RenderLayers>,
    )>,
    quads: Query<&InxCompositeQuad>,
) {
    for (group_entity, mode, group, nested_in, layers) in &groups {
        if *mode != ComposeMode::NeedsRt {
            continue;
        }
        if quads.iter().any(|q| q.0 == group_entity) {
            continue;
        }
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [-0.5f32, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vec![[0.0f32, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        );
        mesh.insert_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]));

        let material = materials.add(InxPartMaterial {
            data: InxPartMaterialUniform {
                tint: Vec3::ONE,
                opacity: 1.0,
                screen_tint: Vec3::ZERO,
                composite: 1,
                emissive_strength: 0.0,
            },
            albedo: None,
            emissive: None,
            bumpmap: None,
            blend_mode: group.blend_mode,
        });
        let mut ec = commands.spawn((
            InxCompositeQuad(group_entity),
            ChildOf(group_entity),
            Mesh2d(meshes.add(mesh)),
            MeshMaterial2d(material),
            Transform::IDENTITY,
            Visibility::default(),
        ));
        // Nested composite: the quad stands in for the whole group, so it inherits
        // the group's own `InComposite` - the outer composite then treats this quad
        // as a regular member (bbox, Z-packing, extract).
        if let Some(outer) = nested_in {
            ec.insert(*outer);
        }
        // RenderLayers doesn't propagate through Bevy hierarchy - the quad needs its
        // own copy of the group's layer for camera filtering.
        if let Some(layers) = layers {
            ec.insert(layers.clone());
        }
    }
}

/// Per-frame: place each composite quad over its group's world bbox and point its
/// material at the RT acquired this frame. Preserves the Z that [`sync_part_z`]
/// wrote (runs after it).
pub fn sync_composite_quads(
    mut commands: Commands,
    mut materials: ResMut<Assets<InxPartMaterial>>,
    mut quads: Query<(
        Entity,
        &InxCompositeQuad,
        &mut GlobalTransform,
        &mut Visibility,
        &MeshMaterial2d<InxPartMaterial>,
    )>,
    groups: Query<(&crate::composite::InxCompositeBbox, &InxCompositeGroup)>,
) {
    for (quad_entity, quad, mut global, mut visibility, material) in &mut quads {
        let Ok((bbox, group)) = groups.get(quad.0) else {
            // Group despawned (e.g. puppet reload) - quad is orphaned, remove it.
            commands.entity(quad_entity).despawn();
            continue;
        };
        let size = bbox.rect.size();
        if bbox.rt.is_none() || size.x <= 0.0 || size.y <= 0.0 {
            *visibility = Visibility::Hidden;
            continue;
        }
        *visibility = Visibility::Visible;
        let center = bbox.rect.center();
        let z = global.translation().z;
        *global = GlobalTransform::from(
            Transform::from_translation(Vec3::new(center.x, center.y, z))
                .with_scale(Vec3::new(size.x, size.y, 1.0)),
        );
        if let Some(mut mat) = materials.get_mut(&material.0) {
            if mat.albedo != bbox.rt {
                mat.albedo = bbox.rt.clone();
            }
            if mat.blend_mode != group.blend_mode {
                mat.blend_mode = group.blend_mode;
            }
        }
    }
}

/// Z band the puppet parts are spread over. Place external sprites at a Z inside the
/// band to interleave them between parts.
#[derive(Resource)]
pub struct InxZBand {
    /// Z of the frontmost part.
    pub base: f32,
    /// Z increment between consecutive draw-order parts.
    pub step: f32,
}

impl Default for InxZBand {
    fn default() -> Self {
        Self {
            base: 0.0,
            step: 0.1,
        }
    }
}

/// Draw order matching Inochi2D: accumulate zsort down the tree, stable-sort
/// descending (ties keep pre-order, as Inochi2D expects) and assign each part a Z by
/// rank inside the band. Writes GlobalTransform directly, after propagation - many
/// parts share the same accumulated zsort, so encoding the raw value in Z would z-fight.
///
/// Composites are atomic Z units: the group occupies a single rank at its own
/// accumulated zsort and its member parts are packed into the open sub-band
/// `(unit_z, unit_z + step/2]`, ordered among themselves by their accumulated zsort.
/// External sprites placed at whole-step Zs therefore cannot interleave between
/// composite children - matching inochi2d's composite semantics (the door sprite cannot slip between iris and cornea).
#[allow(clippy::type_complexity)] // Bevy Query filter types
pub fn sync_part_z(
    band: Res<InxZBand>,
    roots: Query<(Entity, &GlobalTransform), (With<InxPuppetRoot>, Without<Mesh2d>)>,
    children: Query<&Children>,
    zsorts: Query<&InxZSort>,
    groups: Query<(), With<ComposeMode>>,
    tags: Query<&InComposite>,
    mut parts: Query<(Entity, &mut GlobalTransform, Option<&InxCompositeQuad>), With<Mesh2d>>,
) {
    // group entity -> its final quad (NeedsRt composites only).
    let quad_map: HashMap<Entity, Entity> = parts
        .iter()
        .filter_map(|(e, _, quad)| quad.map(|q| (q.0, e)))
        .collect();
    enum Unit {
        Part(Entity),
        Group(Entity),
    }

    for (root, root_global) in roots.iter() {
        // DFS pre-order: collect parts and composite groups with their accumulated
        // zsort. Groups go into a Vec (not a HashMap) so unit order stays
        // deterministic.
        let mut part_list: Vec<(Entity, f32)> = Vec::new();
        let mut group_list: Vec<(Entity, f32)> = Vec::new();
        // Root's own world Z lets multiple puppets in the same scene be separated by
        // the user (Transform on the InxScene root) - the zsort-derived band only
        // orders parts *within* one puppet.
        let root_offset = root_global.translation().z;
        let root_z = zsorts.get(root).map(|z| z.0).unwrap_or(0.0);
        let mut stack = vec![(root, root_z)];
        while let Some((entity, acc)) = stack.pop() {
            if parts.contains(entity) {
                part_list.push((entity, acc));
            }
            if groups.contains(entity) {
                group_list.push((entity, acc));
            }
            if let Ok(ch) = children.get(entity) {
                for child in ch.iter().rev() {
                    let cz = zsorts.get(child).map(|z| z.0).unwrap_or(0.0);
                    stack.push((child, acc + cz));
                }
            }
        }

        // Split parts into standalone units and composite members.
        let group_set: HashSet<Entity> = group_list.iter().map(|(e, _)| *e).collect();
        let mut members: HashMap<Entity, Vec<(Entity, f32)>> = HashMap::default();
        let mut units: Vec<(f32, Unit)> = Vec::new();
        for (entity, acc) in part_list {
            match tags.get(entity) {
                Ok(tag) if group_set.contains(&tag.0) => {
                    members.entry(tag.0).or_default().push((entity, acc));
                }
                _ => units.push((acc, Unit::Part(entity))),
            }
        }
        for (group, acc) in group_list {
            if members.contains_key(&group) {
                units.push((acc, Unit::Group(group)));
            }
        }

        units.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (_, unit)) in units.iter().enumerate() {
            let unit_z = root_offset + band.base + rank as f32 * band.step;
            match unit {
                Unit::Part(entity) => {
                    if let Ok((_, mut global, _)) = parts.get_mut(*entity) {
                        let mut t = global.compute_transform();
                        t.translation.z = unit_z;
                        *global = GlobalTransform::from(t);
                    }
                }
                Unit::Group(group) => {
                    // The composite's final quad (NeedsRt) sits exactly at the
                    // group's atomic rank.
                    if let Some(quad_entity) = quad_map.get(group)
                        && let Ok((_, mut global, _)) = parts.get_mut(*quad_entity)
                    {
                        let mut t = global.compute_transform();
                        t.translation.z = unit_z;
                        *global = GlobalTransform::from(t);
                    }
                    let list = members.get_mut(group).unwrap();
                    list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let intra = band.step * 0.5 / (list.len() as f32 + 1.0);
                    for (i, (entity, _)) in list.iter().enumerate() {
                        if let Ok((_, mut global, _)) = parts.get_mut(*entity) {
                            let mut t = global.compute_transform();
                            t.translation.z = unit_z + (i as f32 + 1.0) * intra;
                            *global = GlobalTransform::from(t);
                        }
                    }
                }
            }
        }
    }
}

/// Parts with a changed deform, excluding masked ones (see [`sync_part_deforms`] for why).
type DeformedUnmaskedParts<'w, 's> = Query<
    'w,
    's,
    (&'static Mesh2d, &'static InxMaterial, &'static InxDeform),
    (Changed<InxDeform>, Without<InxPartMasks>),
>;

/// Rewrites mesh positions when deforms change.
///
/// Masked parts are skipped: their `Mesh` is a clipped rebuild (different vertex count)
/// owned by `sync_mask_clipping`, which re-runs on `Changed<InxDeform>` with the
/// deformed positions. Writing the full unclipped buffer here would mismatch
/// attribute lengths (Bevy truncates to the shortest with the "Vertex_Uv has a different vertex count" warning, corrupting the geometry).
pub fn sync_part_deforms(mut meshes: ResMut<Assets<Mesh>>, parts: DeformedUnmaskedParts) {
    for (mesh2d, mat, deform) in parts.iter() {
        let Some(positions) = part_positions(mat, Some(deform)) else {
            continue;
        };
        if let Some(mut mesh) = meshes.get_mut(&mesh2d.0) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        }
    }
}

/// Resolved mask sources of a masked part.
#[derive(Component)]
pub struct InxPartMasks {
    /// Mask source entities and their mode.
    pub sources: Vec<(Entity, InxMaskMode)>,
}

/// Marker: part already inspected for masks.
#[derive(Component)]
pub struct InxMaskLinked;

type Contour = Vec<[f32; 2]>;

/// Resolves mask source uuids to entities for masked parts.
#[allow(clippy::type_complexity)] // Bevy Query filter types
pub fn link_part_masks(
    mut commands: Commands,
    parts: Query<(Entity, &InxMaterial), (With<Mesh2d>, Without<InxMaskLinked>)>,
    uuids: Query<(Entity, &InxUUID)>,
    materials: Query<&InxMaterial>,
    parents: Query<&ChildOf>,
) {
    if parts.is_empty() {
        return;
    }
    // Keyed by (owning puppet root, node uuid): node uuids are per-file, not
    // globally unique, so a flat uuid map would let a mask on one puppet resolve to
    // a same-uuid node on ANOTHER puppet in a multi-puppet scene.
    let uuid_map: HashMap<(Entity, u32), Entity> = uuids
        .iter()
        .map(|(e, u)| ((crate::root_of(e, &parents), u.0), e))
        .collect();

    for (entity, mat) in parts.iter() {
        commands.entity(entity).insert(InxMaskLinked);
        if mat.masks.is_empty() || mat.mesh.is_none() {
            continue;
        }

        let root = crate::root_of(entity, &parents);
        let mut sources = Vec::new();
        for mask in &mat.masks {
            let Some(&src) = uuid_map.get(&(root, mask.source_uuid)) else {
                continue;
            };
            let has_mesh = materials.get(src).is_ok_and(|m| m.mesh.is_some());
            if has_mesh {
                sources.push((src, mask.mode));
            }
        }

        if !sources.is_empty() {
            commands.entity(entity).insert(InxPartMasks { sources });
        }
    }
}

/// Deformed positions in part-local 2D space.
fn deformed_local(mat: &InxMaterial, deform: Option<&InxDeform>) -> Option<Vec<[f32; 2]>> {
    let mesh = mat.mesh.as_ref()?;
    let origin = mesh.origin;
    Some(
        mesh.vertex_buffer
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d = deform
                    .and_then(|d| d.offsets.get(i))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                [v[0] - origin.x + d[0], v[1] - origin.y + d[1]]
            })
            .collect(),
    )
}

/// Every triangle of the mesh as a CCW contour, optionally transformed.
/// `FillRule::NonZero` then unions them into the exact mesh silhouette - robust
/// against duplicated vertices, seams and overlaps, no boundary topology extraction needed.
fn triangle_contours(
    positions: &[[f32; 2]],
    indices: &[u32],
    rel: Option<&bevy::math::Affine3A>,
    out: &mut Vec<Contour>,
) {
    for tri in indices.chunks_exact(3) {
        let mut c: Contour = tri
            .iter()
            .map(|&i| {
                let p = positions[i as usize];
                match rel {
                    Some(m) => {
                        let t = m.transform_point3(Vec3::new(p[0], p[1], 0.0));
                        [t.x, t.y]
                    }
                    None => p,
                }
            })
            .collect();
        let area = (c[1][0] - c[0][0]) * (c[2][1] - c[0][1])
            - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1]);
        if area == 0.0 {
            continue;
        }
        if area < 0.0 {
            c.reverse();
        }
        out.push(c);
    }
}

/// UV for a clipped vertex: barycentric interpolation inside the original (deformed)
/// triangle that contains it; falls back to the closest triangle.
fn uv_for_point(p: [f32; 2], positions: &[[f32; 2]], uvs: &[[f32; 2]], indices: &[u32]) -> [f32; 2] {
    let mut best_uv = [0.0, 0.0];
    let mut best_score = f32::NEG_INFINITY;
    for tri in indices.chunks_exact(3) {
        let a = positions[tri[0] as usize];
        let b = positions[tri[1] as usize];
        let c = positions[tri[2] as usize];
        let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if det.abs() < f32::EPSILON {
            continue;
        }
        let u = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / det;
        let v = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / det;
        let w = 1.0 - u - v;
        let score = u.min(v).min(w);
        if score > best_score {
            best_score = score;
            let (ua, ub, uc) = (
                uvs[tri[0] as usize],
                uvs[tri[1] as usize],
                uvs[tri[2] as usize],
            );
            best_uv = [
                w * ua[0] + u * ub[0] + v * uc[0],
                w * ua[1] + u * ub[1] + v * uc[1],
            ];
        }
    }
    best_uv
}

/// Inverse of `uv_for_point`: given a point in UV space, find the triangle whose UV
/// coordinates contain it and barycentric-interpolate the LOCAL (deformed) position
/// instead. Position and UV are affine per-triangle, so the UV-space barycentric
/// weights carry over directly.
fn local_for_uv(uv: [f32; 2], positions: &[[f32; 2]], uvs: &[[f32; 2]], indices: &[u32]) -> [f32; 2] {
    let mut best_local = [0.0, 0.0];
    let mut best_score = f32::NEG_INFINITY;
    for tri in indices.chunks_exact(3) {
        let a = uvs[tri[0] as usize];
        let b = uvs[tri[1] as usize];
        let c = uvs[tri[2] as usize];
        let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if det.abs() < f32::EPSILON {
            continue;
        }
        let u = ((uv[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (uv[1] - a[1])) / det;
        let v = ((b[0] - a[0]) * (uv[1] - a[1]) - (uv[0] - a[0]) * (b[1] - a[1])) / det;
        let w = 1.0 - u - v;
        let score = u.min(v).min(w);
        if score > best_score {
            best_score = score;
            let (pa, pb, pc) = (
                positions[tri[0] as usize],
                positions[tri[1] as usize],
                positions[tri[2] as usize],
            );
            // Clamp barycentric weights for points outside every triangle (e.g. a mask contour point just past the render mesh's coarser UV hull)
            // - unclamped extrapolation blows up near thin/near degenerate triangles
            // (tiny `det`), throwing the mapped point wildly outside the mesh.
            let (cu, cv, cw) = (u.max(0.0), v.max(0.0), w.max(0.0));
            let sum = cu + cv + cw;
            let (cu, cv, cw) = if sum > f32::EPSILON {
                (cu / sum, cv / sum, cw / sum)
            } else {
                (0.0, 0.0, 1.0)
            };
            best_local = [
                cw * pa[0] + cu * pb[0] + cv * pc[0],
                cw * pa[1] + cu * pb[1] + cv * pc[1],
            ];
        }
    }
    best_local
}

/// Baked alpha contours (UV space) of a mask source, mapped into local mesh space
/// via `local_for_uv` and optionally transformed like `triangle_contours`. Gives a
/// real alpha silhouette instead of the coarse mesh outline - the source mesh is
/// usually a loose quad around the visible texture.
fn alpha_contours(
    contours: &[Vec<[f32; 2]>],
    positions: &[[f32; 2]],
    uvs: &[[f32; 2]],
    indices: &[u32],
    rel: Option<&bevy::math::Affine3A>,
    out: &mut Vec<Contour>,
) {
    for contour in contours {
        let mut c: Contour = contour
            .iter()
            .map(|&uv| {
                let p = local_for_uv(uv, positions, uvs, indices);
                match rel {
                    Some(m) => {
                        let t = m.transform_point3(Vec3::new(p[0], p[1], 0.0));
                        [t.x, t.y]
                    }
                    None => p,
                }
            })
            .collect();
        if c.len() < 3 {
            continue;
        }
        let area = signed_area_2d(&c);
        if area == 0.0 {
            continue;
        }
        if area < 0.0 {
            c.reverse();
        }
        out.push(c);
    }
}

fn signed_area_2d(c: &Contour) -> f32 {
    let mut sum = 0.0;
    for i in 0..c.len() {
        let a = c[i];
        let b = c[(i + 1) % c.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum * 0.5
}

/// CPU mask clipping: intersect each masked part's geometry with the union of its
/// Mask sources and subtract its Dodge sources, then retriangulate and rewrite the
/// render mesh. Uses the source's baked alpha contour (`mask_contour_uv`) as the
/// mask silhouette when the INR provides one; falls back to the mesh's own triangle
/// outline otherwise (coarser - the source mesh is usually a loose quad around the visible texture).
#[allow(clippy::type_complexity)] // Bevy Query filter types
pub fn sync_mask_clipping(
    mut meshes: ResMut<Assets<Mesh>>,
    masked: Query<(
        Entity,
        &Mesh2d,
        &InxMaterial,
        Option<&InxDeform>,
        &GlobalTransform,
        &InxPartMasks,
    )>,
    sources: Query<(&InxMaterial, Option<&InxDeform>, &GlobalTransform)>,
    // Dirty tracking: a masked part only needs re-clipping when its own
    // geometry/transform/material changed or one of its mask sources did. Without
    // this the boolean overlay + retriangulation + mesh re-upload ran for EVERY
    // masked part EVERY frame, even with the puppet at rest.
    changed: Query<
        (),
        Or<(
            Changed<InxDeform>,
            Changed<GlobalTransform>,
            Changed<InxMaterial>,
            Added<InxPartMasks>,
        )>,
    >,
) {
    for (entity, mesh2d, mat, deform, global, masks) in masked.iter() {
        let dirty = changed.contains(entity)
            || masks.sources.iter().any(|&(src, _)| changed.contains(src));
        if !dirty {
            continue;
        }
        let Some(inx_mesh) = mat.mesh.as_ref() else {
            continue;
        };
        let Some(local) = deformed_local(mat, deform) else {
            continue;
        };

        let mut subj: Vec<Contour> = Vec::new();
        triangle_contours(&local, &inx_mesh.index_buffer, None, &mut subj);
        if subj.is_empty() {
            continue;
        }

        let inv = global.affine().inverse();
        let mut mask_clip: Vec<Contour> = Vec::new();
        let mut dodge_clip: Vec<Contour> = Vec::new();
        for &(src, mode) in &masks.sources {
            let Ok((smat, sdeform, sglobal)) = sources.get(src) else {
                continue;
            };
            let (Some(smesh), Some(slocal)) = (smat.mesh.as_ref(), deformed_local(smat, sdeform))
            else {
                continue;
            };
            let rel = inv * sglobal.affine();
            let out = match mode {
                InxMaskMode::Mask => &mut mask_clip,
                InxMaskMode::Dodge => &mut dodge_clip,
            };
            match &smesh.mask_contour_uv {
                Some(contours) => alpha_contours(
                    contours,
                    &slocal,
                    &smesh.uv_buffer,
                    &smesh.index_buffer,
                    Some(&rel),
                    out,
                ),
                None => triangle_contours(&slocal, &smesh.index_buffer, Some(&rel), out),
            }
        }
        if mask_clip.is_empty() && dodge_clip.is_empty() {
            continue;
        }

        let mut shapes: Shapes<[f32; 2]> = if mask_clip.is_empty() {
            subj.overlay(&dodge_clip, OverlayRule::Difference, FillRule::NonZero)
        } else {
            subj.overlay(&mask_clip, OverlayRule::Intersect, FillRule::NonZero)
        };
        if !mask_clip.is_empty() && !dodge_clip.is_empty() {
            shapes = shapes.overlay(&dodge_clip, OverlayRule::Difference, FillRule::NonZero);
        }

        // A `Mask` is meant as a near-total containment silhouette (the masked part lives inside it)
        // - small concave dips in the mask's own baked contour crossing the part's
        // boundary (e.g. a jaw cusp grazing a mouth part) can split the exact
        // boolean intersect into disjoint pieces, leaving a sliver gap that reads as
        // a false notch once triangulated. When the intersect still recovers ~all of
        // the part's own area, treat the gap as clip-precision noise and keep the
        // un-clipped geometry instead of the fragmented result.
        if !mask_clip.is_empty() && dodge_clip.is_empty() {
            let subj_area: f32 = subj.iter().map(|c| signed_area_2d(c).abs()).sum();
            let clipped_area: f32 = shapes
                .as_slice()
                .iter()
                .flat_map(|shape| shape.iter())
                .map(|path| signed_area_2d(path).abs())
                .sum();
            if subj_area > 0.0 && clipped_area / subj_area >= 0.95 {
                // Still owns the mesh rebuild: restore the FULL deformed geometry
                // (sync_part_deforms skips masked parts, and the mesh may hold a clipped build from a previous frame).
                if let Some(mut mesh) = meshes.get_mut(&mesh2d.0) {
                    let positions: Vec<[f32; 3]> =
                        local.iter().map(|p| [p[0], p[1], 0.0]).collect();
                    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, inx_mesh.uv_buffer.clone());
                    mesh.insert_indices(Indices::U32(inx_mesh.index_buffer.clone()));
                }
                continue;
            }
        }

        let tri = shapes.as_slice().triangulate().to_triangulation::<u32>();
        let positions: Vec<[f32; 3]> = tri.points.iter().map(|p| [p[0], p[1], 0.0]).collect();
        let uvs: Vec<[f32; 2]> = tri
            .points
            .iter()
            .map(|p| uv_for_point(*p, &local, &inx_mesh.uv_buffer, &inx_mesh.index_buffer))
            .collect();

        // A part whose masks cover it entirely clips away to nothing. Writing empty
        // buffers would leave the mesh with no GPU allocation while still being
        // uploaded every frame, so stand in a zero-area triangle: it keeps both
        // buffers allocated and rasterizes to no fragments. The real geometry comes
        // back on the next frame the part is no longer fully covered.
        let (positions, uvs, indices) = if positions.is_empty() || tri.indices.is_empty() {
            (vec![[0.0, 0.0, 0.0]; 3], vec![[0.0, 0.0]; 3], vec![0, 1, 2])
        } else {
            (positions, uvs, tri.indices)
        };

        if let Some(mut mesh) = meshes.get_mut(&mesh2d.0) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
            mesh.insert_indices(Indices::U32(indices));
        }
    }
}

/// Pushes animated opacity/tint into the material asset.
pub fn sync_part_materials(
    mut materials: ResMut<Assets<InxPartMaterial>>,
    parts: Query<(
        Entity,
        &MeshMaterial2d<InxPartMaterial>,
        &InxMaterial,
        Option<&InComposite>,
    )>,
    changed_parts: Query<(), (Changed<InxMaterial>, With<MeshMaterial2d<InxPartMaterial>>)>,
    groups: Query<(&InxMaterial, &ComposeMode), With<InxCompositeGroup>>,
    changed_groups: Query<(), (Changed<InxMaterial>, With<InxCompositeGroup>)>,
) {
    if changed_parts.is_empty() && changed_groups.is_empty() {
        return;
    }
    for (entity, handle, mat, member) in parts.iter() {
        let group_changed = member.is_some_and(|c| changed_groups.contains(c.0));
        if !changed_parts.contains(entity) && !group_changed {
            continue;
        }
        let group = per_child_blend_group(member, &groups);
        let (data, _) = part_material_values(mat, group);
        if let Some(mut material) = materials.get_mut(&handle.0) {
            material.data = data;
        }
    }
}

/// The enclosing group's material, but only when the group renders as
/// `PerChildBlend` - the only mode where composite fields fold into each child's own
/// material (children proven disjoint by the exporter hint).
fn per_child_blend_group<'a>(
    member: Option<&InComposite>,
    groups: &'a Query<(&InxMaterial, &ComposeMode), With<InxCompositeGroup>>,
) -> Option<&'a InxMaterial> {
    member
        .and_then(|c| groups.get(c.0).ok())
        .filter(|(_, mode)| **mode == ComposeMode::PerChildBlend)
        .map(|(gmat, _)| gmat)
}

/// Strength gated on the texture actually existing: the fallback image bound for a
/// missing emissive texture is white, so a nonzero strength without a texture would
/// glow the whole part.
fn part_emissive_strength(mat: &InxMaterial) -> f32 {
    if mat.texture_emissive.is_some() {
        mat.emissive_strength
    } else {
        0.0
    }
}

fn part_material_values(
    mat: &InxMaterial,
    group: Option<&InxMaterial>,
) -> (InxPartMaterialUniform, BlendMode) {
    match group {
        Some(g) => (
            InxPartMaterialUniform {
                tint: mat.tint * g.tint,
                opacity: mat.opacity * g.opacity,
                // Screen tints compose like screen blending: 1-(1-a)(1-b).
                screen_tint: Vec3::ONE
                    - (Vec3::ONE - mat.screen_tint) * (Vec3::ONE - g.screen_tint),
                composite: 0,
                emissive_strength: part_emissive_strength(mat),
            },
            g.blend_mode,
        ),
        None => (
            InxPartMaterialUniform {
                tint: mat.tint,
                opacity: mat.opacity,
                screen_tint: mat.screen_tint,
                composite: 0,
                emissive_strength: part_emissive_strength(mat),
            },
            mat.blend_mode,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_child_blend_combines_group_and_part_values() {
        let part = InxMaterial {
            tint: Vec3::new(1.0, 0.8, 0.6),
            opacity: 0.9,
            screen_tint: Vec3::new(0.2, 0.0, 0.0),
            blend_mode: BlendMode::Normal,
            ..Default::default()
        };
        let group = InxMaterial {
            tint: Vec3::new(0.5, 1.0, 1.0),
            opacity: 0.5,
            screen_tint: Vec3::new(0.0, 0.5, 0.0),
            blend_mode: BlendMode::Multiply,
            ..Default::default()
        };

        let (data, blend) = part_material_values(&part, Some(&group));
        assert_eq!(blend, BlendMode::Multiply);
        assert!((data.opacity - 0.45).abs() < 1e-6);
        assert!(data.tint.abs_diff_eq(Vec3::new(0.5, 0.8, 0.6), 1e-6));
        // screen: 1-(1-a)(1-b) per channel
        assert!(data.screen_tint.abs_diff_eq(Vec3::new(0.2, 0.5, 0.0), 1e-6));

        let (solo, solo_blend) = part_material_values(&part, None);
        assert_eq!(solo_blend, BlendMode::Normal);
        assert!((solo.opacity - 0.9).abs() < 1e-6);
        assert!(solo.tint.abs_diff_eq(part.tint, 1e-6));
        assert!(solo.screen_tint.abs_diff_eq(part.screen_tint, 1e-6));
    }
}
