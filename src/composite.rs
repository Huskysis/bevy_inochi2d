//! Composite groups for the Mesh2d renderer (data-first).
//!
//! Most composites need no render target at all: [`classify`] collapses them into a
//! plain Z-band ([`ComposeMode::Normal`]) when the "over" operator's associativity
//! allows it, so members just draw in-place like regular parts. Only the rare
//! composite that genuinely needs isolation (e.g. `Multiply` with overlapping children, or an explicit mask-affecting mode)
//! escalates to [`ComposeMode::NeedsRt`]: its subtree renders into an offscreen
//! render target (from [`CompositeRtPool`]), and the result is drawn as a single
//! [`InxCompositeQuad`] with the group's blend mode, opacity and tint.
//!
//! Everything for both paths - bbox computation, RT pool, visibility isolation, the
//! render-graph pass - lives in this one module.

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::RenderLayers,
    image::Image,
    platform::collections::HashMap,
    prelude::*,
    render::{
        Extract,
        render_asset::RenderAssets,
        render_resource::{
            Extent3d, LoadOp, Operations, RenderPassColorAttachment, RenderPassDescriptor,
            StoreOp, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::RenderContext,
        texture::GpuImage,
    },
};

use crate::{BlendMode, InxDeform, InxMaterial};

/// Bucket sizes used by [`CompositeRtPool`]. Powers of two between 64 and 2048:
/// small facial composites fit the low buckets, larger groups the higher ones, with
/// 2048 reserved for edge-case large groups.
pub const COMPOSITE_RT_BUCKETS: [u32; 6] = [64, 128, 256, 512, 1024, 2048];

/// Maximum RT side in pixels. Bbox sizes above this are clamped with a one-shot warning
/// (see [`CompositeRtPool::bucket_for`]).
pub const COMPOSITE_RT_MAX: u32 = 2048;

/// Shared pool of square RGBA8 render targets used by the composite render-graph pass.
///
/// Each frame:
/// 1. [`CompositeRtPool::acquire`] pops a free handle from the matching bucket (or allocates one)
///    and tracks it in `in_flight`.
/// 2. The composite pass renders into it; the final quad samples it.
/// 3. [`CompositeRtPool::release_frame`] (run after the main 2D pass) moves every
///    in-flight handle back to its bucket for reuse next frame.
///
/// Handles are shared across puppets so an idle composite in puppet A's RT can be
/// reused by puppet B's composite of the same bucket size next frame.
#[derive(Resource, Default)]
pub struct CompositeRtPool {
    free: HashMap<u32, Vec<Handle<Image>>>,
    in_flight: Vec<(Handle<Image>, u32)>,
    warned_oversize: bool,
    /// Compositing space the pooled textures were created for. Pooled handles are
    /// dropped when it changes, since it decides their format.
    srgb_compositing: bool,
}

impl CompositeRtPool {
    /// Pick the smallest bucket that can hold a `side`×`side` bbox. Sides above
    /// [`COMPOSITE_RT_MAX`] are clamped to it.
    pub fn bucket_for(&mut self, side: u32) -> u32 {
        if side > COMPOSITE_RT_MAX && !self.warned_oversize {
            self.warned_oversize = true;
            bevy::log::warn!(
                "composite bbox side {side}px exceeds RT cap {COMPOSITE_RT_MAX}px - clamping (further warnings suppressed)"
            );
        }
        let target = side.min(COMPOSITE_RT_MAX).max(COMPOSITE_RT_BUCKETS[0]);
        *COMPOSITE_RT_BUCKETS
            .iter()
            .find(|b| **b >= target)
            .unwrap_or(COMPOSITE_RT_BUCKETS.last().unwrap())
    }

    /// Acquire a render-target handle large enough for `side` pixels. The returned
    /// handle remains in flight until `release_frame` is called.
    pub fn acquire(&mut self, side: u32, images: &mut Assets<Image>) -> Handle<Image> {
        let bucket = self.bucket_for(side);
        let srgb = self.srgb_compositing;
        let handle = self
            .free
            .get_mut(&bucket)
            .and_then(|v| v.pop())
            .unwrap_or_else(|| images.add(make_rt(bucket, srgb)));
        self.in_flight.push((handle.clone(), bucket));
        handle
    }

    /// Set the compositing space the pool creates textures for, dropping pooled
    /// handles when it changes so no texture outlives its format.
    pub fn set_srgb_compositing(&mut self, srgb: bool) {
        if self.srgb_compositing != srgb {
            self.srgb_compositing = srgb;
            self.free.clear();
        }
    }

    /// Texture format the pool currently hands out.
    pub fn format(&self) -> TextureFormat {
        rt_format(self.srgb_compositing)
    }

    /// Move every in-flight handle back to its bucket. Call once per frame after the
    /// composite RTs have been sampled by the main pass.
    pub fn release_frame(&mut self) {
        for (handle, bucket) in self.in_flight.drain(..) {
            self.free.entry(bucket).or_default().push(handle);
        }
    }
}

/// Whether a camera's compositing space means gamma-encoded blending.
///
/// Composite RTs and their synthetic views follow the camera so a composite blends the
/// same way the rest of the scene does. With several cameras the first one decides.
fn scene_srgb_compositing(space: Option<&bevy::camera::CompositingSpace>) -> bool {
    matches!(space, Some(bevy::camera::CompositingSpace::Srgb))
}

/// RT format for a compositing space. Gamma-encoded compositing stores already-encoded
/// values, so it needs a non-sRGB format: an sRGB view would encode them a second time
/// on write.
fn rt_format(srgb_compositing: bool) -> TextureFormat {
    if srgb_compositing {
        TextureFormat::Rgba8Unorm
    } else {
        TextureFormat::Rgba8UnormSrgb
    }
}

fn make_rt(side: u32, srgb_compositing: bool) -> Image {
    let size = Extent3d {
        width: side,
        height: side,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        rt_format(srgb_compositing),
        RenderAssetUsages::default(),
    );
    // COPY_SRC enables debug readback (RT dump via bevy::render::gpu_readback).
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC
        | TextureUsages::RENDER_ATTACHMENT;
    image
}

/// Composite group configuration, attached to the group entity.
///
/// Mirrors the composite fields the loader writes into `InxMaterial`, lifted to a
/// dedicated component so the Mesh2d renderer can query composites without dragging `InxMaterial` around.
#[derive(Component, Debug, Clone, Reflect)]
pub struct InxCompositeGroup {
    /// Blend mode for the entire group.
    pub blend_mode: BlendMode,
    /// Group global opacity.
    pub opacity: f32,
    /// Additive tint applied to the entire group.
    pub tint: Vec3,
    /// Group screen tint.
    pub screen_tint: Vec3,
    /// Alpha threshold used by masks scoped to this group. The CPU clipping path
    /// does not consume it; kept for future texture-based masks.
    pub mask_threshold: f32,
    /// Padding (in puppet units) added around the computed bbox before quantising to
    /// a pow2 RT bucket.
    pub padding: f32,
    /// Z used for the final composite quad in the main pass.
    pub zsort: f32,
}

impl Default for InxCompositeGroup {
    fn default() -> Self {
        Self {
            blend_mode: BlendMode::Normal,
            opacity: 1.0,
            tint: Vec3::ONE,
            screen_tint: Vec3::ZERO,
            mask_threshold: 0.5,
            padding: 8.0,
            zsort: 0.0,
        }
    }
}

/// How a composite group is realised by the Mesh2d renderer.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum ComposeMode {
    /// Identity composite (Normal blend, opacity 1, identity tints). "Over" is
    /// associative, so drawing the children directly, packed into an atomic Z band,
    /// is pixel-identical to offscreen compositing.
    Grouping,
    /// Group blend proven safe to apply per child (exporter verified the children never overlap each other; `compose_hint` in the INR).
    PerChildBlend,
    /// Requires real offscreen compositing (render-target fallback path).
    NeedsRt,
}

/// Exporter-baked overlap verdict for a non-identity composite, read from the INR
/// `compose_hint` field. `None` = identity composite or a file written by an older exporter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum InxComposeHint {
    /// Children proven pairwise disjoint at every sampled pose.
    ChildrenDisjoint,
    /// Children overlap at some pose, or the analysis was inconclusive.
    ChildrenOverlap,
}

/// Classify a composite from its group parameters plus the exporter's overlap proof.
/// Non-identity composites downgrade from `NeedsRt` to `PerChildBlend` only when the
/// INR `compose_hint` proves the children disjoint.
pub fn classify(group: &InxCompositeGroup, hint: Option<InxComposeHint>) -> ComposeMode {
    const EPS: f32 = 1e-6;
    let identity = group.blend_mode == BlendMode::Normal
        && (group.opacity - 1.0).abs() < EPS
        && group.tint.abs_diff_eq(Vec3::ONE, EPS)
        && group.screen_tint.abs_diff_eq(Vec3::ZERO, EPS);
    if identity {
        ComposeMode::Grouping
    } else if hint == Some(InxComposeHint::ChildrenDisjoint) {
        ComposeMode::PerChildBlend
    } else {
        ComposeMode::NeedsRt
    }
}

/// Upgrades a freshly spawned `Grouping`/`PerChildBlend` composite to `NeedsRt` when
/// any descendant part uses `ClipToLower`/`SliceFromLower`. Both modes flatten
/// children directly into the shared framebuffer; that is pixel-identical to
/// offscreen compositing IF children only read their own src color (Normal, per-child blend proven disjoint).
/// ClipToLower and SliceFromLower instead read *destination* alpha, which under
/// flattening is whatever was drawn earlier in the WHOLE scene (e.g. opaque skin under a mouth composite),
/// not the group-local context Inochi2d intends. Real-model example: Mouth/Eye
/// composites classified `Grouping` with `ClipToLower` children (teeth, tongue, iris, eye lights) -
/// clip reads scene dst-alpha instead of clipping to nothing, so content draws unclipped.
pub fn upgrade_composite_mode_for_dst_reading_children(
    mut groups: Query<(Entity, &mut ComposeMode), Added<InxCompositeGroup>>,
    members: Query<(&InComposite, &crate::InxMaterial)>,
) {
    for (group_entity, mut mode) in &mut groups {
        if *mode == ComposeMode::NeedsRt {
            continue;
        }
        let has_dst_reader = members.iter().any(|(tag, mat)| {
            tag.0 == group_entity
                && matches!(
                    mat.blend_mode,
                    BlendMode::ClipToLower | BlendMode::SliceFromLower
                )
        });
        if has_dst_reader {
            bevy::log::debug!(
                "composite '{group_entity:?}' upgraded {:?} -> NeedsRt: child uses ClipToLower/SliceFromLower (dst-alpha read breaks flatten associativity)",
                *mode
            );
            *mode = ComposeMode::NeedsRt;
        }
    }
}

/// Per-frame bbox of the composite's subtree and the RT it has been rendered into.
/// Written by the bbox system, consumed by the render-graph pass and by the final-quad transform sync.
#[derive(Component, Debug, Default, Clone, Reflect)]
pub struct InxCompositeBbox {
    /// Screen-space bbox of the composite's subtree this frame.
    pub rect: Rect,
    /// Acquired render target, when the group is `NeedsRt`.
    #[reflect(ignore)]
    pub rt: Option<Handle<Image>>,
    /// Side of the acquired RT in pixels (the pool bucket).
    pub rt_side: u32,
}

/// Marker placed on every descendant of a composite group, pointing at the group
/// entity. Propagated at spawn time (recursive walk handles MeshGroup intermediaries without special-casing).
#[derive(Component, Debug, Clone, Copy, Reflect)]
pub struct InComposite(pub Entity);

/// Render layer for parts that only exist inside a composite RT. The main camera
/// renders layer 0, so members of a `NeedsRt` composite move here so they are never
/// drawn directly, only through the composite's quad. The synthetic composite view
/// lists them explicitly in its `RenderVisibleEntities`, bypassing layer filtering.
pub const COMPOSITE_ONLY_LAYER: usize = 31;

/// Marks an entity as already moved to [`COMPOSITE_ONLY_LAYER`] by
/// [`isolate_needs_rt_children`]. A dedicated marker (not "has no `RenderLayers` yet")
/// because a puppet spawned with a caller-provided `RenderLayers` (e.g. `InxScene` + `RenderLayers::layer(1)` for render-to-texture, see `examples/rtt.rs`)
/// already carries one at spawn - gating on "layers.is_none()" would then never
/// isolate NeedsRt composite children, so they'd draw twice (straight onto that layer AND through the composite quad).
#[derive(Component, Debug, Clone, Copy, Reflect)]
pub struct InxRtIsolated;

/// Update system: move every member of a `NeedsRt` composite off to
/// [`COMPOSITE_ONLY_LAYER`], overwriting whatever `RenderLayers` it spawned with.
/// Without this the children draw twice - straight onto the screen AND multiplied
/// through the composite quad.
#[allow(clippy::type_complexity)] // Bevy Query filter types
pub fn isolate_needs_rt_children(
    mut commands: Commands,
    groups: Query<&ComposeMode>,
    // `With<Mesh2d>` (not `InxMaterial`): a nested composite's own quad has no
    // `InxMaterial` but must be isolated the same way when its group is itself a
    // member of an outer NeedsRt composite.
    members: Query<(Entity, &InComposite), (With<bevy::mesh::Mesh2d>, Without<InxRtIsolated>)>,
) {
    for (entity, tag) in &members {
        if groups.get(tag.0) == Ok(&ComposeMode::NeedsRt) {
            commands
                .entity(entity)
                .insert((RenderLayers::layer(COMPOSITE_ONLY_LAYER), InxRtIsolated));
        }
    }
}

/// Custom visibility system (runs in `VisibilitySystems::CheckVisibility`): no
/// main-world camera renders [`COMPOSITE_ONLY_LAYER`], so Bevy's check leaves these
/// parts globally invisible and mesh extraction would skip them. Mark them visible
/// so they reach the render world, where only the synthetic composite view draws them.
pub fn force_composite_member_visibility(
    groups: Query<(&ComposeMode, &InheritedVisibility)>,
    mut members: Query<(&InComposite, &mut ViewVisibility), With<bevy::mesh::Mesh2d>>,
) {
    use bevy::camera::visibility::SetViewVisibility;
    for (tag, mut vis) in &mut members {
        // Gate on the group's own InheritedVisibility too: without this, a hidden
        // puppet (or a hidden composite ancestor) still showed its NeedsRt members -
        // this system forced them visible unconditionally, bypassing the hierarchy's
        // Hidden state entirely.
        if let Ok((mode, inherited)) = groups.get(tag.0)
            && *mode == ComposeMode::NeedsRt
            && inherited.get()
        {
            vis.set_visible();
        }
    }
}

/// Per-frame system: recompute every composite's world-space bbox by aggregating
/// descendant Part vertices (after deforms) and padding it by `group.padding`. Must
/// run after the param/anim pass writes deforms.
///
/// A nested composite (its own group entity tagged `InComposite(outer)`, see `auto_spawn::spawn_node_recursive`)
/// folds its own (already padded) rect into the outer group's accumulation instead
/// of the outer walking its grandchildren's raw vertices directly - the outer only
/// ever sees its direct members plus one rect per nested child, matching how the
/// render pass treats a nested composite as an opaque quad. Resolved by relaxation
/// (a handful of passes - nesting in practice is 1-2 levels deep, but this handles arbitrary depth without a topological sort).
///
/// `Rect::EMPTY` (= `min == max == 0`) is left in place when a group has no
/// renderable descendants - the render pass treats that as a no-op.
pub fn update_composite_bbox(
    mut groups: Query<(Entity, &InxCompositeGroup, &mut InxCompositeBbox, Option<&InComposite>)>,
    members: Query<(
        &InComposite,
        &GlobalTransform,
        &InxMaterial,
        Option<&InxDeform>,
    )>,
) {
    let mut accum: HashMap<Entity, (Vec2, Vec2)> = HashMap::default();

    for (tag, gtf, mat, deform) in &members {
        let Some(mesh) = &mat.mesh else {
            continue;
        };
        let xf = gtf.compute_transform();
        let entry = accum
            .entry(tag.0)
            .or_insert((Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)));

        // Local positions match `mesh2d::part_positions`: (v - origin + deform).
        let origin = mesh.origin;
        for (i, vb) in mesh.vertex_buffer.iter().enumerate() {
            let off = deform
                .and_then(|d| d.offsets.get(i))
                .copied()
                .unwrap_or([0.0, 0.0]);
            let local = Vec3::new(vb[0] - origin.x + off[0], vb[1] - origin.y + off[1], 0.0);
            let world = xf.transform_point(local).truncate();
            entry.0 = entry.0.min(world);
            entry.1 = entry.1.max(world);
        }
    }

    // Nested group -> (outer group, own padding) for the relaxation pass.
    let nesting: Vec<(Entity, Entity, f32)> = groups
        .iter()
        .filter_map(|(entity, group, _, tag)| tag.map(|t| (entity, t.0, group.padding)))
        .filter(|(_, outer, _)| groups.contains(*outer))
        .collect();

    for _ in 0..nesting.len().max(1) {
        let mut changed = false;
        for &(inner, outer, pad) in &nesting {
            let Some(&(min, max)) = accum.get(&inner) else {
                continue;
            };
            if !min.x.is_finite() {
                continue;
            }
            let padded = (min - Vec2::splat(pad), max + Vec2::splat(pad));
            let entry = accum
                .entry(outer)
                .or_insert((Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)));
            let new_min = entry.0.min(padded.0);
            let new_max = entry.1.max(padded.1);
            if new_min != entry.0 || new_max != entry.1 {
                changed = true;
            }
            entry.0 = new_min;
            entry.1 = new_max;
        }
        if !changed {
            break;
        }
    }

    for (group_entity, group, mut bbox, _) in &mut groups {
        let Some((min, max)) = accum.get(&group_entity).copied() else {
            bbox.rect = Rect::EMPTY;
            continue;
        };
        if !min.x.is_finite() || !min.y.is_finite() {
            bbox.rect = Rect::EMPTY;
            continue;
        }
        let pad = Vec2::splat(group.padding);
        bbox.rect = Rect {
            min: min - pad,
            max: max + pad,
        };
    }
}

/// Per-frame system (render-target fallback): give every `NeedsRt` composite with a
/// non-empty bbox a render target from [`CompositeRtPool`], stored in
/// `InxCompositeBbox::rt`. Runs after [`update_composite_bbox`].
///
/// `release_frame` at the top returns last frame's handles to their buckets, so a
/// stable composite reuses the same RT every frame.
pub fn acquire_composite_rts(
    mut pool: ResMut<CompositeRtPool>,
    mut images: ResMut<Assets<Image>>,
    mut groups: Query<(Entity, &ComposeMode, &mut InxCompositeBbox, Option<&Name>)>,
    cameras: Query<Option<&bevy::camera::CompositingSpace>, With<Camera>>,
    mut last_bucket: Local<HashMap<Entity, u32>>,
) {
    pool.set_srgb_compositing(scene_srgb_compositing(cameras.iter().next().flatten()));
    pool.release_frame();
    for (entity, mode, mut bbox, name) in &mut groups {
        if *mode != ComposeMode::NeedsRt {
            continue;
        }
        let size = bbox.rect.size();
        if size.x <= 0.0 || size.y <= 0.0 {
            bbox.rt = None;
            continue;
        }
        let side = size.x.max(size.y).ceil() as u32;
        let bucket = pool.bucket_for(side);
        bbox.rt = Some(pool.acquire(side, &mut images));
        bbox.rt_side = bucket;
        if last_bucket.get(&entity) != Some(&bucket) {
            let label = name.map(|n| n.as_str()).unwrap_or("<unnamed>");
            bevy::log::debug!(
                "composite RT '{label}' (entity {entity:?}) bbox side={side}px -> bucket {bucket}px"
            );
            last_bucket.insert(entity, bucket);
        }
    }
}

/// Snapshot of one `NeedsRt` composite copied into the render world each frame (render-target fallback).
#[derive(Clone, Debug)]
pub struct ExtractedComposite {
    /// Main-world group entity (stable key across frames).
    pub group_entity: Entity,
    /// Nesting depth (count of ancestor composite groups, any mode). Used to render
    /// deepest-first: a nested `NeedsRt` composite's quad samples its RT as a
    /// texture in the outer composite's pass, so the inner RT must already be
    /// rendered when the outer's pass runs.
    pub depth: u32,
    /// Blend mode for the entire group.
    pub blend_mode: BlendMode,
    /// Group global opacity.
    pub opacity: f32,
    /// Additive tint applied to the entire group.
    pub tint: Vec3,
    /// Group screen tint.
    pub screen_tint: Vec3,
    /// Z used for the final composite quad in the main pass.
    pub zsort: f32,
    /// World-space subtree bbox (padded); `Rect::EMPTY` = nothing to draw.
    pub bbox: Rect,
    /// Render target acquired by [`acquire_composite_rts`]; `None` = skip.
    pub rt: Option<Handle<Image>>,
    /// RT side in pixels (pool bucket), for the synthetic view's viewport.
    pub rt_side: u32,
    /// Subtree's renderable descendants as (render-world, main-world) pairs.
    pub children: Vec<(Entity, Entity)>,
}

/// Render-world resource rebuilt every frame by [`extract_composites`].
#[derive(Resource, Default)]
pub struct ExtractedComposites(pub Vec<ExtractedComposite>);

/// ExtractSchedule system: copy every `NeedsRt` composite (group params, bbox, tagged descendants) into [`ExtractedComposites`].
#[allow(clippy::type_complexity)]
pub fn extract_composites(
    mut extracted: ResMut<ExtractedComposites>,
    groups: Extract<
        Query<(
            Entity,
            &InxCompositeGroup,
            &ComposeMode,
            &InxCompositeBbox,
            Option<&InComposite>,
        )>,
    >,
    members: Extract<
        Query<(
            Entity,
            Option<&bevy::render::sync_world::RenderEntity>,
            &InComposite,
            &GlobalTransform,
        )>,
    >,
) {
    // Nested-composite ancestry among ALL groups (any mode), so a NeedsRt group
    // nested a level deeper than its nearest NeedsRt ancestor still gets the right
    // depth (e.g. NeedsRt inside a plain Grouping inside another NeedsRt).
    let parent_of: HashMap<Entity, Entity> = groups
        .iter()
        .filter_map(|(entity, _, _, _, tag)| tag.map(|t| (entity, t.0)))
        .collect();
    let depth_of = |mut e: Entity| -> u32 {
        let mut depth = 0;
        let mut seen = 0;
        while let Some(&parent) = parent_of.get(&e) {
            depth += 1;
            e = parent;
            // Cycle guard - ancestry should be a DAG, but never hang on bad data.
            seen += 1;
            if seen > 64 {
                break;
            }
        }
        depth
    };

    extracted.0.clear();
    for (group_entity, group, mode, bbox, _) in groups.iter() {
        if *mode != ComposeMode::NeedsRt {
            continue;
        }
        // Collect children sorted by ascending Z so the Transparent2d phase draws
        // them back-to-front (painter's order within the RT).
        let mut children: Vec<(Entity, Entity, f32)> = members
            .iter()
            .filter(|(_, _, tag, _)| tag.0 == group_entity)
            .map(|(main, render, _, gt)| {
                let z = gt.translation().z;
                (render.map(|r| r.id()).unwrap_or(Entity::PLACEHOLDER), main, z)
            })
            .collect();
        children.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let children: Vec<(Entity, Entity)> =
            children.into_iter().map(|(r, m, _)| (r, m)).collect();
        extracted.0.push(ExtractedComposite {
            group_entity,
            depth: depth_of(group_entity),
            blend_mode: group.blend_mode,
            opacity: group.opacity,
            tint: group.tint,
            screen_tint: group.screen_tint,
            zsort: group.zsort,
            bbox: bbox.rect,
            rt: bbox.rt.clone(),
            rt_side: bbox.rt_side,
            children,
        });
    }
    // Deepest-first: a nested NeedsRt composite's RT must be rendered before any
    // ancestor composite's pass samples it through the nested quad.
    extracted.0.sort_by_key(|c| std::cmp::Reverse(c.depth));
}

/// Marker on the final quad that draws a `NeedsRt` composite's RT into the main
/// pass, pointing at the group entity. The quad is a standalone unit-square Mesh2d;
/// `mesh2d::sync_composite_quads` sizes and places it over the group bbox and
/// `mesh2d::sync_part_z` gives it the group's atomic Z rank.
#[derive(Component, Debug, Clone, Copy, Reflect)]
pub struct InxCompositeQuad(pub Entity);

/// Render-world component on a synthetic composite view, pointing at the main-world group entity.
#[derive(Component, Debug, Clone, Copy)]
pub struct CompositeViewOf(pub Entity);

/// Render-world map: main-world group entity -> its synthetic view entity. Views are
/// retained across frames and despawned when the composite stops being extracted.
#[derive(Resource, Default)]
pub struct CompositeViewEntities(pub HashMap<Entity, Entity>);

/// Empty schedule assigned to every synthetic composite view.
///
/// Composite views carry an [`ExtractedCamera`](bevy::render::camera::ExtractedCamera) so
/// Bevy's Mesh2d specialization treats them as views, which also makes `camera_driver`
/// pick them up and run their schedule. They are rendered by [`composite_pass`], not by
/// a camera schedule, so the one they point at does nothing.
#[derive(bevy::ecs::schedule::ScheduleLabel, Debug, Hash, PartialEq, Eq, Clone)]
pub struct InxCompositeViewSchedule;

/// ExtractSchedule system (after [`extract_composites`] and after Bevy's `extract_core_2d_camera_phases`, whose retain would drop our phases):
/// keep one synthetic [`ExtractedView`] per extracted composite, with an
/// orthographic projection tightly framing the bbox, plus everything Bevy's Mesh2d
/// machinery expects from a view:
///
/// - `MainEntity` + `Msaa` - `check_views_need_specialization` (view key).
/// - `Camera2d` + `Tonemapping`- `prepare_mesh2d_view_bind_groups` filters
///   `With<Camera2d>`; without the bind group the draw commands panic.
/// - `RenderVisibleEntities` - drives `specialize_/queue_material2d_meshes`.
/// - Transparent2d/Opaque2d/AlphaMask2d phases registered for the retained view
///   (queue bails unless all three exist).
///
/// Bevy's `prepare_view_uniforms` then writes view uniforms for it like for any
/// camera view, and the standard specialize/queue/batch systems fill the
/// Transparent2d phase that [`CompositePassNode`] renders.
#[allow(clippy::too_many_arguments)] // Bevy system: each phase resource is its own param
pub fn queue_composite_views(
    mut commands: Commands,
    mut views: ResMut<CompositeViewEntities>,
    extracted: Res<ExtractedComposites>,
    mut transparent_phases: ResMut<
        bevy::render::render_phase::ViewSortedRenderPhases<
            bevy::core_pipeline::core_2d::Transparent2d,
        >,
    >,
    mut opaque_phases: ResMut<
        bevy::render::render_phase::ViewBinnedRenderPhases<bevy::core_pipeline::core_2d::Opaque2d>,
    >,
    mut alpha_mask_phases: ResMut<
        bevy::render::render_phase::ViewBinnedRenderPhases<
            bevy::core_pipeline::core_2d::AlphaMask2d,
        >,
    >,
    mut dirty_specializations: ResMut<bevy::render::camera::DirtySpecializations>,
    scene_cameras: Extract<Query<Option<&bevy::camera::CompositingSpace>, With<Camera>>>,
) {
    use bevy::camera::{CameraOutputMode, ClearColorConfig, MsaaWriteback};
    use bevy::core_pipeline::tonemapping::Tonemapping;
    use bevy::ecs::schedule::ScheduleLabel;
    use bevy::prelude::Camera2d;
    use bevy::render::batching::gpu_preprocessing::GpuPreprocessingMode;
    use bevy::render::camera::ExtractedCamera;
    use bevy::render::sync_world::MainEntity;
    use bevy::render::view::{
        ColorGrading, ExtractedView, RenderVisibleEntities, RenderVisibleEntitiesClass,
        RetainedViewEntity,
    };

    let compositing_space = scene_cameras.iter().next().flatten().copied();
    let rt_format = rt_format(scene_srgb_compositing(compositing_space.as_ref()));

    let mut seen: Vec<Entity> = Vec::new();
    for (index, c) in extracted.0.iter().enumerate() {
        if c.rt.is_none() {
            continue;
        }
        let size = c.bbox.size();
        if size.x <= 0.0 || size.y <= 0.0 {
            continue;
        }
        let center = c.bbox.center();
        // View sits in front of the puppet Z band looking down -Z; children land in
        // [0, 2000) depth.
        let world_from_view =
            GlobalTransform::from(Transform::from_translation(center.extend(1000.0)));
        let clip_from_view = Mat4::orthographic_rh(
            -size.x / 2.0,
            size.x / 2.0,
            -size.y / 2.0,
            size.y / 2.0,
            0.0,
            2000.0,
        );
        let retained_view_entity = RetainedViewEntity::new(c.group_entity.into(), None, 0);
        let view = ExtractedView {
            retained_view_entity,
            clip_from_view,
            world_from_view,
            clip_from_world: None,
            // Must match the RT the composite pass renders into (see `make_rt`),
            // since it feeds the pipeline specialization key.
            target_format: rt_format,
            viewport: UVec4::new(0, 0, c.rt_side, c.rt_side),
            color_grading: ColorGrading::default(),
            invert_culling: false,
        };
        // B7: all children, already sorted ascending-Z from extract_composites.
        // Bevy's specialize/queue walk these incrementally, so the view is marked
        // dirty below to force a full re-specialize + re-queue every frame.
        let mut entities: Vec<(Entity, MainEntity)> = c
            .children
            .iter()
            .map(|(render, main)| (*render, MainEntity::from(*main)))
            .collect();
        entities.sort_unstable_by_key(|(_, main)| *main);
        let mut visible = RenderVisibleEntities::default();
        visible.classes.insert(
            std::any::TypeId::of::<bevy::mesh::Mesh2d>(),
            RenderVisibleEntitiesClass {
                entities_cpu_culling: entities,
                ..Default::default()
            },
        );
        dirty_specializations.views.insert(retained_view_entity);

        transparent_phases.prepare_for_new_frame(retained_view_entity);
        if let Some(phase) = transparent_phases.get_mut(&retained_view_entity) {
            phase.items.clear();
        }
        opaque_phases.prepare_for_new_frame(retained_view_entity, GpuPreprocessingMode::None);
        alpha_mask_phases.prepare_for_new_frame(retained_view_entity, GpuPreprocessingMode::None);

        let view_entity = *views
            .0
            .entry(c.group_entity)
            .or_insert_with(|| commands.spawn_empty().id());
        commands.entity(view_entity).insert((
            view,
            CompositeViewOf(c.group_entity),
            MainEntity::from(c.group_entity),
            Msaa::Off,
            Tonemapping::None,
            Camera2d,
            // `check_views_need_specialization` only builds a view key for views that
            // carry an `ExtractedCamera`; without a key nothing gets specialized or
            // queued for this view. No render target and an empty schedule keep
            // `camera_driver` from drawing it as a camera.
            ExtractedCamera {
                target: None,
                physical_viewport_size: Some(UVec2::splat(c.rt_side)),
                physical_target_size: Some(UVec2::splat(c.rt_side)),
                viewport: None,
                schedule: InxCompositeViewSchedule.intern(),
                // Distinct per view, far from the orders scene cameras use.
                // Bevy warns when two active cameras share (order, target).
                // These views all carry the same empty target.
                // A scene camera's target goes empty when its window closes.
                // Orders near zero would collide with it during shutdown.
                // Inert value: the schedule they point at does nothing.
                order: isize::MIN + index as isize,
                output_mode: CameraOutputMode::Skip,
                msaa_writeback: MsaaWriteback::Off,
                clear_color: ClearColorConfig::None,
                sorted_camera_index_for_target: 0,
                exposure: 1.0,
                hdr: false,
                // Children must encode their output exactly like the rest of the
                // scene: the quad passes the RT through without re-encoding.
                compositing_space,
            },
            visible,
        ));
        seen.push(c.group_entity);
    }
    views.0.retain(|group, view_entity| {
        let keep = seen.contains(group);
        if !keep {
            commands.entity(*view_entity).despawn();
        }
        keep
    });
}

/// Render-world cache of one depth texture view per RT bucket size. The transparent
/// mesh2d pipeline is specialized with a `Depth32Float` attachment, so the composite
/// pass must bind one even though composites never write depth.
#[derive(Resource, Default)]
pub struct CompositeDepthTextures(pub HashMap<u32, bevy::render::render_resource::TextureView>);

/// Render system (PrepareResources): make sure a depth texture exists for every
/// bucket size used by an extracted composite this frame.
pub fn prepare_composite_depth_textures(
    mut depths: ResMut<CompositeDepthTextures>,
    extracted: Res<ExtractedComposites>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
) {
    use bevy::render::render_resource::{TextureDescriptor, TextureViewDescriptor};
    for c in &extracted.0 {
        if c.rt.is_none() || depths.0.contains_key(&c.rt_side) {
            continue;
        }
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("inx_composite_depth"),
            size: Extent3d {
                width: c.rt_side,
                height: c.rt_side,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        depths
            .0
            .insert(c.rt_side, texture.create_view(&TextureViewDescriptor::default()));
    }
}

/// Render system for the NeedsRt fallback: clears each extracted composite's RT
/// to transparent black (blend no-op for uncovered texels) and renders the
/// composite's Transparent2d phase - filled by Bevy's standard queue systems via the
/// synthetic view - into it (B6).
///
/// Runs once per frame in the root render schedule, before any camera schedule, so
/// composite RTs are ready when their quads sample them in the main 2D pass.
pub fn composite_pass(
    world: &World,
    extracted: Res<ExtractedComposites>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    view_entities: Res<CompositeViewEntities>,
    transparent_phases: Res<
        bevy::render::render_phase::ViewSortedRenderPhases<
            bevy::core_pipeline::core_2d::Transparent2d,
        >,
    >,
    depths: Res<CompositeDepthTextures>,
    mut render_context: RenderContext,
) {
    {
        use bevy::render::view::RetainedViewEntity;

        for composite in &extracted.0 {
            let Some(rt) = &composite.rt else { continue };
            let Some(gpu) = gpu_images.get(rt) else { continue };
            let Some(depth_view) = depths.0.get(&composite.rt_side) else {
                continue;
            };
            let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("inx_composite_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &gpu.texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(LinearRgba::NONE.into()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(
                    bevy::render::render_resource::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        // Reverse-Z like the main 2D pass: clear to 0.0.
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(0.0),
                            store: StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    },
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let retained = RetainedViewEntity::new(composite.group_entity.into(), None, 0);
            let (Some(view_entity), Some(phase)) = (
                view_entities.0.get(&composite.group_entity).copied(),
                transparent_phases.get(&retained),
            ) else {
                continue;
            };
            if !phase.items.is_empty()
                && let Err(err) = phase.render(&mut pass, world, view_entity)
            {
                bevy::log::error!("composite pass render error: {err:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_composite_is_grouping_regardless_of_hint() {
        let group = InxCompositeGroup::default();
        assert_eq!(classify(&group, None), ComposeMode::Grouping);
        assert_eq!(
            classify(&group, Some(InxComposeHint::ChildrenOverlap)),
            ComposeMode::Grouping
        );
        assert_eq!(
            classify(&group, Some(InxComposeHint::ChildrenDisjoint)),
            ComposeMode::Grouping
        );
    }

    #[test]
    fn non_identity_needs_disjoint_proof_for_per_child_blend() {
        let group = InxCompositeGroup {
            blend_mode: BlendMode::Multiply,
            ..Default::default()
        };
        assert_eq!(classify(&group, None), ComposeMode::NeedsRt);
        assert_eq!(
            classify(&group, Some(InxComposeHint::ChildrenOverlap)),
            ComposeMode::NeedsRt
        );
        assert_eq!(
            classify(&group, Some(InxComposeHint::ChildrenDisjoint)),
            ComposeMode::PerChildBlend
        );
    }

    #[test]
    fn non_identity_opacity_or_tint_also_classify() {
        let translucent = InxCompositeGroup {
            opacity: 0.5,
            ..Default::default()
        };
        assert_eq!(
            classify(&translucent, Some(InxComposeHint::ChildrenDisjoint)),
            ComposeMode::PerChildBlend
        );
        let tinted = InxCompositeGroup {
            tint: Vec3::new(1.0, 0.5, 0.5),
            ..Default::default()
        };
        assert_eq!(classify(&tinted, None), ComposeMode::NeedsRt);
    }

    #[test]
    fn nested_composite_bbox_folds_into_outer() {
        use std::sync::Arc;

        let mut world = World::new();

        let mesh = Arc::new(crate::InxMesh {
            vertex_buffer: vec![[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
            uv_buffer: vec![[0.0, 0.0]; 4],
            index_buffer: vec![0, 1, 2, 0, 2, 3],
            origin: Vec2::ZERO,
            mask_contour_uv: None,
        });

        let outer = world
            .spawn((
                InxCompositeGroup {
                    padding: 5.0,
                    ..Default::default()
                },
                InxCompositeBbox::default(),
            ))
            .id();
        let inner = world
            .spawn((
                InxCompositeGroup {
                    padding: 2.0,
                    ..Default::default()
                },
                InxCompositeBbox::default(),
                InComposite(outer),
            ))
            .id();
        // Direct member of outer, far from the inner's region (300 units away).
        world.spawn((
            InComposite(outer),
            GlobalTransform::from(Transform::from_xyz(300.0, 0.0, 0.0)),
            InxMaterial {
                mesh: Some(mesh.clone()),
                ..Default::default()
            },
        ));
        // Direct member of inner only.
        world.spawn((
            InComposite(inner),
            GlobalTransform::IDENTITY,
            InxMaterial {
                mesh: Some(mesh),
                ..Default::default()
            },
        ));

        let mut schedule = Schedule::default();
        schedule.add_systems(update_composite_bbox);
        schedule.run(&mut world);

        let outer_rect = world.get::<InxCompositeBbox>(outer).unwrap().rect;
        let inner_rect = world.get::<InxCompositeBbox>(inner).unwrap().rect;

        // Inner: member spans [-10,10] padded by 2 -> [-12,12].
        assert_eq!(inner_rect.min, Vec2::splat(-12.0));
        assert_eq!(inner_rect.max, Vec2::splat(12.0));

        // Outer must contain BOTH its direct member (around x=300) and the inner's
        // own (padded) rect - not just its direct member.
        assert!(outer_rect.min.x <= -12.0, "outer lost the nested rect: {outer_rect:?}");
        assert!(outer_rect.max.x >= 310.0, "outer lost its direct member: {outer_rect:?}");
    }

    #[test]
    fn bucket_for_clamps_oversize_to_max() {
        let mut pool = CompositeRtPool::default();
        assert_eq!(pool.bucket_for(4096), COMPOSITE_RT_MAX);
        assert_eq!(pool.bucket_for(2049), COMPOSITE_RT_MAX);
        assert_eq!(pool.bucket_for(COMPOSITE_RT_MAX), COMPOSITE_RT_MAX);
        assert_eq!(pool.bucket_for(1), COMPOSITE_RT_BUCKETS[0]);
    }
}
