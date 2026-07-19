//! Inochi2D puppet renderer built on Bevy's native `Mesh2d`/`Material2d` ecosystem.

#![warn(missing_docs)]

/// Multi-layer animation controller, default-pose application and per-frame param evaluation.
pub mod animation;
/// Spawns a puppet's node tree as ECS entities from a loaded [`InxPuppet`].
pub mod auto_spawn;
/// Composite groups: Z-band collapsing and the offscreen render-target fallback.
pub mod composite;
/// 2D grid interpolation (linear/cubic/stepped) shared by params and animations.
pub mod grid_interpolation;
pub use inochi2d_parser::inr;
/// `.inr` `AssetLoader`.
pub mod inr_loader;
/// `.inx`/`.inp` `AssetLoader` (feature `inx`).
#[cfg(feature = "inx")]
pub mod loader;
/// `Mesh2d` sync systems: deforms, materials, mask clipping, Z order.
pub mod mesh2d;
/// Bevy `Plugin`s that register this crate's systems.
pub mod plugin;
/// Pendulum/spring-pendulum physics driving params.
pub mod simple_physics;

use std::sync::Arc;

use bevy::{
    camera::visibility::{VisibilityClass, add_visibility_class},
    platform::collections::HashMap,
    prelude::*,
    render::{
        extract_component::ExtractComponent,
        render_resource::{BlendComponent, BlendState},
    },
};

/// Re-exports the crate's common types for glob import.
pub mod prelude {
    pub use crate::{
        AnimationLayer, BlendMode, FadeState, InxAnimation, InxAnimationController,
        InxAnimationLane, InxBasePose, InxBinding, InxBindingValues, InxDefaultPose, InxDeform,
        InxInterpolation, InxKeyframe, InxMask, InxMaskMode, InxMaterial, InxMergeMode, InxMesh,
        InxMeshGroupWarp, InxMeshWrap, InxNode, InxNodeType, InxParam, InxParamState, InxProp,
        InxPuppet, InxPuppetRoot, InxResolvedBindings, InxScene, InxUUID, InxZSort, MeshWrap,
        MgChildMap, SimplePhysicsConfig,
        composite::{
            self, ComposeMode, InComposite, InxComposeHint, InxCompositeBbox, InxCompositeGroup,
        },
        grid_interpolation::{DeformAccum, NodeAccum, cubic_hermite, ease_in_out, lerp},
        mesh2d::{InxMesh2dPlugin, InxPartMasks},
        plugin::{Inochi2dAnimationPlugin, Inochi2dCorePlugin, Inochi2dPlugin},
        simple_physics::{
            InxPhysicsState, InxPuppetPhysicsEnabled, InxSimplePhysics, PhysicsEnabled,
            PhysicsMapMode, PhysicsModel,
        },
    };
    #[cfg(feature = "inx")]
    pub use inochi2d_parser::prelude::{MaskMode, Transform as InxTransform};
}

/// Command component to spawn a puppet as a scene.
///
/// Usage:
/// ```ignore
/// let my_puppet_handle: Handle<InxPuppet> = ...;
/// commands.spawn(InxScene {
///     puppet: my_puppet_handle,
///     transform: Transform::from_xyz(0.0, 0.0, 0.0),
///     animation: true,
///     default_pose: false,
/// });
/// ```
#[derive(Component)]
pub struct InxScene {
    /// Handle to the loaded InxPuppet asset.
    pub puppet: Handle<InxPuppet>,
    /// Root transform of the instance.
    pub transform: Transform,
    /// If true, inserts InxAnimationController with all animations.
    pub animation: bool,
    /// If true, applies the default pose (params at their authored value) without
    /// registering `Inochi2dAnimationPlugin`'s full loop. See [`InxDefaultPose`] / `animation::apply_default_pose`.
    pub default_pose: bool,
}

/// Marker inserted on the root when `InxScene::default_pose` is true. Makes
/// `animation::apply_default_pose` resolve this puppet's params to its authored pose
/// every frame, without needing `InxAnimationPlugin` (which brings the full animation/physics loop).
#[derive(Component, Debug, Clone, Copy, Reflect)]
pub struct InxDefaultPose;

/// Loaded puppet asset (via `InrLoader`/`InxLoader`): node tree, params, animations
/// and textures, plus lookups by name.
#[derive(Debug, Asset, TypePath)]
pub struct InxPuppet {
    /// Root-level nodes of the tree.
    pub nodes: Vec<InxNode>,
    /// All params, load order.
    pub params: Vec<Handle<InxParam>>,
    /// All animations, load order.
    pub animations: Vec<Handle<InxAnimation>>,
    /// All textures, indexed as authored.
    pub textures: Vec<Handle<Image>>,

    /// Nodes indexed by name, for lookup at spawn/scripting time.
    pub named_nodes: HashMap<Box<str>, InxNode>,
    // pub named_mesh: HashMap<Box<str>, Handle<Mesh>>, pub named_material:
    // HashMap<Box<str>, Handle<Material2d>>,
    /// Params indexed by name.
    pub named_params: HashMap<Box<str>, Handle<InxParam>>,
    /// Animations indexed by name.
    pub named_animations: HashMap<Box<str>, Handle<InxAnimation>>,

    // Metadata
    /// Authored metadata.
    pub meta: InxMeta,
    /// Global physics configuration.
    pub physics: InxPhysics,
}

/// Authored puppet metadata (name, rigger, license, etc), as it comes in the INR/INX
/// - doesn't affect rendering.
#[derive(Debug, Clone)]
pub struct InxMeta {
    /// Descriptive name of the puppet.
    pub name: String,
    /// Inochi2D format version used.
    pub version: String,
    /// Rigger name.
    pub rigger: String,
    /// Artist name.
    pub artist: String,
    /// Usage and distribution rights.
    pub rights: String,
    /// Model copyright.
    pub copyright: String,
    /// URL to usage license.
    pub license_url: String,
    /// Creator contact information.
    pub contact: String,
    /// Visual reference or link of the model.
    pub reference: String,
    /// Texture ID for UI thumbnail.
    pub thumbnail_id: u32,
    /// If true, preserves pixels during render (no smoothing).
    pub preserve_pixels: bool,
}

/// Global physics parameters of the puppet (used by `simple_physics_system`).
#[derive(Debug, Clone, Component)]
pub struct InxPhysics {
    /// Pixels-per-meter scale used by simulated nodes.
    pub pixels_per_meter: f32,
    /// Gravity applied to simulated nodes.
    pub gravity: f32,
}

/// Node of the puppet tree.
#[derive(Asset, Debug, TypePath, Component, Clone)]
pub struct InxNode {
    /// Authored UUID (see [`InxUUID`] for the spawned-entity equivalent).
    pub uuid: u32,
    /// Readable node name.
    pub name: Box<str>,
    /// Node type (Part, Composite, MeshGroup, etc).
    pub node_type: InxNodeType,
    /// Material data, populated for `Part` nodes.
    pub material: Option<InxMaterial>,
    /// Mesh of a `MeshGroup` node (its warp lattice). Parts keep their mesh inside
    /// `material`; this is only populated for MeshGroups.
    pub mesh: Option<Arc<InxMesh>>,
    /// MeshGroup only: `dynamic_deformation` - true = warp children at runtime from
    /// their deformed vertices (recompute + replace); false = static rest-pose warp. See `InxMeshGroupWarp`.
    pub mesh_group_dynamic: bool,
    /// Composite only: `propagateMeshGroup`. When false, an ancestor MeshGroup's
    /// warp does NOT cross into this composite's children (it is a propagation barrier).
    /// Default true. Non-composite nodes: unused.
    pub composite_propagate_meshgroup: bool,
    /// Local transform (position, rotation, scale).
    pub transform: Transform,
    /// Authored depth relative to siblings; see [`InxZSort`].
    pub zsort: f32,
    /// If false, the node and its children are not rendered.
    pub enabled: bool,
    /// Physics config, populated for `SimplePhysics` nodes.
    pub physics_data: Option<SimplePhysicsConfig>,
    /// Baked compositing hint, populated for `Composite` nodes.
    pub compose_hint: Option<composite::InxComposeHint>,
    /// Child nodes.
    pub children: Vec<InxNode>,
}

/// Type of node in the puppet tree.
#[derive(Debug, Clone, Copy, Component, PartialEq, Eq, Reflect)]
pub enum InxNodeType {
    /// Visual node with mesh and textures.
    Part,
    /// Visual container with blend mode and opacity.
    Composite,
    /// Node defining a mask for clipping descendants.
    Mask,
    /// Group of meshes with dynamic deformation.
    MeshGroup,
    /// Camera node (parsed but no runtime effect).
    Camera,
    /// Simulated physics node (pendulum/spring).
    SimplePhysics,
    /// Generic node with no specific data.
    Generic,
}

/// Authored UUID of the INR/INX node - stable across reloads, unlike Bevy's
/// `Entity`. Used to resolve param bindings.
#[derive(Default, Clone, Copy, Component, ExtractComponent, Debug, Reflect)]
#[require(VisibilityClass)]
#[component(on_add = add_visibility_class::<InxUUID>)]
pub struct InxUUID(pub u32);

/// Handle to a generic Bevy `Mesh` (not Inochi2D-specific).
#[derive(Clone, Default, Component, Reflect)]
pub struct MeshWrap(pub Handle<Mesh>);

/// Handle to an [`InxMesh`] (raw vertex/UV data, pre-conversion to Bevy `Mesh`).
#[derive(Clone, Default, Component, Reflect)]
pub struct InxMeshWrap(pub Handle<InxMesh>);

/// Raw mesh data of a part, as it comes in the INR (vertices in local space, without deform).
/// `sync_part_deforms`/`attach_part_meshes` convert it to the Bevy `Mesh` the renderer consumes.
#[derive(Asset, Debug, Clone, Component, Default, Reflect)]
pub struct InxMesh {
    /// Vertex positions, local space, pre-deform.
    pub vertex_buffer: Vec<[f32; 2]>,
    /// UV coordinates, one pair per vertex.
    pub uv_buffer: Vec<[f32; 2]>,
    /// Triangle indices into `vertex_buffer`.
    pub index_buffer: Vec<u32>,
    /// Texture offset (NOT the node transform).
    pub origin: Vec2,
    /// Baked alpha silhouette in UV space (0..1), when this part is used as a mask
    /// source and the INR carries a baked contour for it. `None` falls back to the
    /// mesh's own triangle outline as the mask shape.
    #[reflect(ignore)]
    pub mask_contour_uv: Option<Vec<Vec<[f32; 2]>>>,
}

/// 2D material for Inochi2d.
#[derive(Asset, Debug, Clone, Component, Default, Reflect)]
#[require(InxUUID, InxZSort, Transform, Visibility)]
pub struct InxMaterial {
    /// Part geometry.
    #[reflect(ignore)]
    pub mesh: Option<Arc<InxMesh>>,
    /// Albedo texture.
    pub texture_albedo: Option<Handle<Image>>,
    /// Emissive texture.
    pub texture_emissive: Option<Handle<Image>>,
    /// Bump-map texture.
    pub texture_bumpmap: Option<Handle<Image>>,
    /// Texture indices [albedo, emissive, bump] as authored.
    pub textures: [u32; 3],
    /// Additive RGB tint.
    pub tint: Vec3,
    /// Screen tint.
    pub screen_tint: Vec3,
    /// Global opacity.
    pub opacity: f32,
    /// Emission strength.
    pub emissive_strength: f32,
    /// Alpha clipping threshold.
    pub mask_threshold: f32,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Masks clipping this part.
    pub masks: Vec<InxMask>,
}

/// Authored depth of the node, relative to its siblings - NOT the final
/// `Transform.z`. `sync_part_z` (mesh2d.rs) accumulates these values through the
/// hierarchy and converts them into the real `Transform.z` (`z = -zsort`, in the sense that higher zsort = further back).
#[derive(Debug, Clone, Copy, Default, Component, PartialEq, Reflect)]
pub struct InxZSort(pub f32);

/// Pseudo-sprite: external content rendered *inside* the puppet pipeline.
///
/// Spawn it as a child of any puppet node entity and it draws as a textured quad
/// participating in the global zsort (e.g. an item in the hand, between the arm and the hair).
/// The entity is regular ECS - transforms, queries, collisions all work; it draws
/// through the same `Mesh2d`/`Material2d` path as puppet parts, so it depth-sorts
/// and batches with them natively.
///
/// ```ignore
/// // find the hand node by name, then:
/// commands.entity(hand).with_child((
///     InxProp {
///         texture: asset_server.load("sword.png"),
///         size: Vec2::new(64.0, 200.0),
///         ..default()
///     },
///     InxZSort(0.01), // slightly in front of the hand
/// ));
/// ```
#[derive(Component, Clone, Debug, Reflect)]
#[require(InxUUID, InxZSort, InxNodeType = InxNodeType::Part, Transform, Visibility)]
pub struct InxProp {
    /// Texture drawn on the quad.
    pub texture: Handle<Image>,
    /// Quad size in puppet units.
    pub size: Vec2,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Opacity.
    pub opacity: f32,
    /// Additive RGB tint.
    pub tint: Vec3,
}

impl Default for InxProp {
    fn default() -> Self {
        Self {
            texture: Handle::default(),
            size: Vec2::splat(100.0),
            blend_mode: BlendMode::Normal,
            opacity: 1.0,
            tint: Vec3::ONE,
        }
    }
}

/// Bumped on the puppet root whenever the node structure changes (prop added/removed/reparented, or an `InxProp`'s `ChildOf`/`InxZSort` changes - see `sync_props`).
/// Currently write-only: nothing in the Mesh2d renderer reads it yet (each part is its own `Mesh2d` entity, so there's no cached draw list to invalidate).
/// Kept for a future incremental-rebuild path.
#[derive(Debug, Clone, Copy, Default, Component, Reflect)]
pub struct InxStructureVersion(pub u32);

static NEXT_PROP_UUID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0x8000_0000);

pub(crate) fn root_of(mut entity: Entity, parents: &Query<&ChildOf>) -> Entity {
    while let Ok(child_of) = parents.get(entity) {
        entity = child_of.parent();
    }
    entity
}

// Bevy system: many query params + inherently complex Query filter types.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn sync_props(
    mut commands: Commands,
    changed: Query<(Entity, &InxProp), Or<(Added<InxProp>, Changed<InxProp>)>>,
    added: Query<(), Added<InxProp>>,
    // Reparenting or zsort changes count as a structural change too.
    reparented: Query<Entity, (With<InxProp>, Or<(Changed<ChildOf>, Changed<InxZSort>)>)>,
    mut removed: RemovedComponents<InxProp>,
    parents: Query<&ChildOf>,
    mut versions: Query<&mut InxStructureVersion>,
    roots: Query<Entity, (With<InxUUID>, Without<ChildOf>)>,
) {
    let mut dirty_roots: Vec<Entity> = Vec::new();
    let mut dirty_all = false;

    for (entity, prop) in changed.iter() {
        let hw = prop.size.x * 0.5;
        let hh = prop.size.y * 0.5;
        let mesh = InxMesh {
            vertex_buffer: vec![[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
            uv_buffer: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            index_buffer: vec![0, 1, 2, 0, 2, 3],
            origin: Vec2::ZERO,
            mask_contour_uv: None,
        };
        let mut e = commands.entity(entity);
        e.insert(InxMaterial {
            mesh: Some(Arc::new(mesh)),
            texture_albedo: Some(prop.texture.clone()),
            texture_emissive: None,
            texture_bumpmap: None,
            textures: [0; 3],
            tint: prop.tint,
            screen_tint: Vec3::ZERO,
            opacity: prop.opacity,
            emissive_strength: 0.0,
            mask_threshold: 0.5,
            blend_mode: prop.blend_mode,
            masks: Vec::new(),
        });
        if added.contains(entity) {
            e.insert(InxUUID(
                NEXT_PROP_UUID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ));
        }
        dirty_roots.push(root_of(entity, &parents));
    }

    for entity in reparented.iter() {
        dirty_roots.push(root_of(entity, &parents));
    }

    for entity in removed.read() {
        if parents.get(entity).is_ok() {
            dirty_roots.push(root_of(entity, &parents));
        } else {
            // Entity despawned - can no longer find its root.
            dirty_all = true;
        }
    }

    if dirty_all {
        dirty_roots.extend(roots.iter());
    }

    dirty_roots.sort_unstable();
    dirty_roots.dedup();
    for root in dirty_roots {
        if let Ok(mut v) = versions.get_mut(root) {
            v.0 = v.0.wrapping_add(1);
        } else {
            commands.entity(root).insert(InxStructureVersion(1));
        }
    }
}

/// Blend mode of a part/composite. Each variant maps to a wgpu `BlendState` via
/// [`BlendMode::blend_state`]; the complex modes
/// (`ColorBurn`/`HardLight`/`SoftLight`/`Difference`/`Exclusion`/`Inverse`) have no
/// exact fixed-function blend-state equivalent and fall back to `Normal`.
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, Reflect)]
pub enum BlendMode {
    /// Standard alpha compositing.
    #[default]
    Normal,
    /// Multiply colors (darken).
    Multiply,
    /// Screen blend (lighten, light effect).
    Screen,
    /// Overlay (combines Multiply and Screen).
    Overlay,
    /// Darken (only darker pixels).
    Darken,
    /// Lighten (only lighter pixels).
    Lighten,
    /// Color dodge (lightens selectively).
    ColorDodge,
    /// Linear dodge (lightens linearly).
    LinearDodge,
    /// Add (sums colors, glow effect).
    Add,
    /// Color burn (darkens selectively). Falls back to `Normal` (no exact fixed-function equivalent).
    ColorBurn,
    /// Hard light (strong contrast). Falls back to `Normal`.
    HardLight,
    /// Soft light (soft contrast). Falls back to `Normal`.
    SoftLight,
    /// Subtract (subtracts colors).
    Subtract,
    /// Difference (absolute color difference). Falls back to `Normal`.
    Difference,
    /// Exclusion (soft difference). Falls back to `Normal`.
    Exclusion,
    /// Inverse (inverts based on overlapping color factor). Falls back to `Normal`.
    Inverse,
    /// DestinationIn (keeps only pixels where destination exists).
    DestinationIn,
    /// ClipToLower (clipping respecting transparency, against lower content).
    ClipToLower,
    /// SliceFromLower (inverse of ClipToLower, cuts by lower content).
    SliceFromLower,
}

#[cfg(feature = "inx")]
impl From<inochi2d_parser::prelude::BlendMode> for BlendMode {
    fn from(mode: inochi2d_parser::prelude::BlendMode) -> Self {
        use inochi2d_parser::prelude::BlendMode::*;
        match mode {
            Normal => Self::Normal,
            Multiply => Self::Multiply,
            Screen => Self::Screen,
            Overlay => Self::Overlay,
            Darken => Self::Darken,
            Lighten => Self::Lighten,
            ColorDodge => Self::ColorDodge,
            LinearDodge => Self::LinearDodge,
            Add => Self::Add,
            ColorBurn => Self::ColorBurn,
            HardLight => Self::HardLight,
            SoftLight => Self::SoftLight,
            Subtract => Self::Subtract,
            Difference => Self::Difference,
            Exclusion => Self::Exclusion,
            Inverse => Self::Inverse,
            DestinationIn => Self::DestinationIn,
            ClipToLower => Self::ClipToLower,
            SliceFromLower => Self::SliceFromLower,
        }
    }
}

impl BlendMode {
    /// All variants, used by `InxPartMaterialKey`/specialize to generate a
    /// specialized pipeline per mode.
    pub const ALL: [Self; 19] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::LinearDodge,
        BlendMode::Add,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
        BlendMode::Subtract,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Inverse,
        BlendMode::DestinationIn,
        BlendMode::ClipToLower,
        BlendMode::SliceFromLower,
    ];
    /// wgpu fixed-function blend state equivalent to the Inochi2D mode. Modes
    /// without an exact fixed-function equivalent fall to `_ =>` (same as `Normal`).
    pub fn blend_state(&self) -> BlendState {
        use bevy::render::render_resource::{BlendFactor::*, BlendOperation::*};
        match self {
            // Normal: src * srcα + dst * (1 - srcα)
            BlendMode::Normal =>
            // BlendState::REPLACE, BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            {
                BlendState {
                    color: BlendComponent {
                        src_factor: One,
                        dst_factor: OneMinusSrcAlpha,
                        operation: Add,
                    },
                    alpha: BlendComponent {
                        src_factor: One,
                        dst_factor: OneMinusSrcAlpha,
                        operation: Add,
                    },
                }
            }

            // Multiply: src * dst + dst * (1 - srcα)
            BlendMode::Multiply => BlendState {
                color: BlendComponent {
                    src_factor: Dst,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },

            // Screen: src + dst * (1 - src)
            BlendMode::Screen => BlendState {
                color: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusSrc,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },

            // Overlay: src * dstα + dst * (1 - srcα) Approx. (Photoshop-like)
            BlendMode::Overlay => BlendState {
                color: BlendComponent {
                    src_factor: SrcAlpha,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },

            // Darken: min(src, dst)
            BlendMode::Darken => BlendState {
                color: BlendComponent {
                    src_factor: One,
                    dst_factor: One,
                    operation: Min,
                },
                alpha: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },
            // Lighten: max(src, dst)
            BlendMode::Lighten => BlendState {
                color: BlendComponent {
                    src_factor: One,
                    dst_factor: One,
                    operation: Max,
                },
                alpha: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },

            // ColorDodge: src * dst + dst
            BlendMode::ColorDodge => BlendState {
                color: BlendComponent {
                    src_factor: Dst,
                    dst_factor: One,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusDstAlpha,
                    operation: Add,
                },
            },

            // LinearDodge / Add: src + dst
            BlendMode::Add | BlendMode::LinearDodge => BlendState {
                color: BlendComponent {
                    src_factor: SrcAlpha,
                    dst_factor: One,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: Zero,
                    dst_factor: One,
                    operation: Add,
                },
            },

            // Subtract: dst - src
            BlendMode::Subtract => BlendState {
                color: BlendComponent {
                    src_factor: One,
                    dst_factor: One,
                    operation: ReverseSubtract,
                },
                alpha: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },

            // DestinationIn: dst * srcα (mask operation)
            BlendMode::DestinationIn => BlendState {
                color: BlendComponent {
                    src_factor: Zero,
                    dst_factor: SrcAlpha,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: Zero,
                    dst_factor: SrcAlpha,
                    operation: Add,
                },
            },

            // DestinationOut: dstα- src BlendMode::DestinationOut => BlendState {
            // color: BlendComponent { src_factor: DstAlpha, dst_factor: Zero,
            // operation: Add, }, alpha: BlendComponent { src_factor: DstAlpha,
            // dst_factor: Zero, operation: Add, }, },

            // ClipToLower: src * dstα + dst * (1 - srcα) Clips source to destination
            // alpha (shows only where dst exists)
            BlendMode::ClipToLower => BlendState {
                color: BlendComponent {
                    src_factor: DstAlpha,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: DstAlpha,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },

            // SliceFromLower: dst * (1 - srcα) - src * dstα Cuts out source from
            // destination
            BlendMode::SliceFromLower => BlendState {
                color: BlendComponent {
                    src_factor: Zero,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: Zero,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },

            // SourceIn: srcα- dst BlendMode::SourceIn => BlendState { color:
            // BlendComponent { src_factor: DstAlpha, dst_factor: Zero, operation:
            // Add, }, alpha: BlendComponent { src_factor: DstAlpha, dst_factor:
            // Zero, operation: Add, }, },

            // SourceOut: src - dstα BlendMode::SourceOut => BlendState { color:
            // BlendComponent { src_factor: OneMinusDstAlpha, dst_factor: Zero,
            // operation: Zero, }, alpha: BlendComponent { src_factor:
            // OneMinusDstAlpha, dst_factor: Zero, operation: Add, }, },

            // Complex modes - fallback to Normal (need multi-pass for accurate)
            // TODO: Implement via shader or multi-pass BlendMode::ColorBurn |
            // BlendMode::HardLight | BlendMode::SoftLight | BlendMode::Difference |
            // BlendMode::Exclusion | BlendMode::Inverse
            _ => BlendState {
                color: BlendComponent {
                    src_factor: SrcAlpha,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
                alpha: BlendComponent {
                    src_factor: One,
                    dst_factor: OneMinusSrcAlpha,
                    operation: Add,
                },
            },
        }
    }
}

/// Reference to a source node that clips this part (CPU clipping, no stencil).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Reflect)]
pub struct InxMask {
    /// UUID of the node providing the mask shape.
    pub source_uuid: u32,
    /// How the mask is applied.
    pub mode: InxMaskMode,
}

/// `Mask` clips (shows only inside the source); `Dodge` clips the inverse (shows only outside the source).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Reflect)]
pub enum InxMaskMode {
    /// Standard clipping (shows only inside the mask).
    #[default]
    Mask,
    /// Dodge/inverse (shows only outside the mask).
    Dodge,
}

#[cfg(feature = "inx")]
impl From<inochi2d_parser::owned::MaskMode> for InxMaskMode {
    fn from(mode: inochi2d_parser::owned::MaskMode) -> Self {
        use inochi2d_parser::owned::MaskMode::*;
        match mode {
            Mask => Self::Mask,
            Dodge => Self::Dodge,
        }
    }
}

/// Deformation offsets per vertex, updated every frame. Accumulated from multiple
/// params (`evaluate_params`/`apply_default_pose`) and applied on the CPU by
/// rewriting `Mesh::ATTRIBUTE_POSITION` (`sync_part_deforms`, mesh2d.rs) - no shader-side deform.
#[derive(Component, Debug, Clone, Default, Reflect)]
pub struct InxDeform {
    /// Offset [dx, dy] per vertex. Length = mesh vertex count.
    pub offsets: Vec<[f32; 2]>,
}

/// Static barycentric mapping from a `MeshGroup`'s lattice to one descendant part's
/// vertices, built once at spawn from rest poses (INR space).
#[derive(Debug, Clone, Reflect)]
pub struct MgChildMap {
    /// The child part entity this mapping warps.
    pub entity: Entity,
    /// Per child-vertex: lattice triangle vertex indices + barycentric weights.
    /// `None` = vertex outside every lattice triangle beyond tolerance (left unwarped).
    #[reflect(ignore)]
    pub map: Vec<Option<([u32; 3], [f32; 3])>>,
    /// Inverse linear part (no translation) of the group->child transform chain:
    /// brings a group-space offset into the child's local space.
    pub inv_linear: [[f32; 2]; 2],
    /// Forward linear part (child->group), inverse of `inv_linear`: maps a
    /// child-local deform into group space. Used by the dynamic path to shift the
    /// barycentric query point by the child's own deform.
    pub fwd_linear: [[f32; 2]; 2],
    /// Rest query point of each child vertex in group-local space
    /// (`part_to_group.apply(local)`), so the dynamic path re-locates the deformed
    /// point without recomputing the affine each frame.
    pub rest_query: Vec<[f32; 2]>,
}

/// On a `MeshGroup` entity: how its Deform bindings warp descendant parts. Groups
/// without Deform bindings never get this component (zero cost).
#[derive(Component, Debug, Clone, Reflect)]
pub struct InxMeshGroupWarp {
    /// Barycentric mapping for each descendant part.
    pub children: Vec<MgChildMap>,
    /// Group lattice rest positions in group-local space (origin-adjusted), +
    /// triangle indices - for the dynamic path's per-frame re-lookup.
    pub lattice_rest: Vec<[f32; 2]>,
    /// Lattice triangle indices into `lattice_rest`.
    pub tris: Vec<[u32; 3]>,
    /// `dynamic_deformation` of the source MeshGroup: true = warp children from
    /// their deformed vertices each frame (recompute + replace), false = static rest-pose additive warp.
    pub dynamic: bool,
}

/// Animatable puppet parameter (2D grid: `is_vec2` decides whether the Y axis is used).
/// `bindings` are the nodes/fields this param drives.
#[derive(Asset, Debug, Component, Reflect)]
pub struct InxParam {
    /// Global unique identifier of the parameter.
    pub uuid: u32,
    /// Readable name.
    pub name: String,
    /// If true, is 2D vector (X, Y); if false, is scalar.
    pub is_vec2: bool,
    /// Minimum allowed value (x, y if vec2).
    pub min: [f32; 2],
    /// Maximum allowed value (x, y if vec2).
    pub max: [f32; 2],
    /// Default value at load (x, y if vec2).
    pub defaults: [f32; 2],
    /// Points on X and Y axes for discrete interpolation.
    pub axis_points: [Vec<f32>; 2],
    /// How multiple bindings affecting the same target are combined.
    pub merge_mode: InxMergeMode,
    /// Nodes/properties this parameter affects.
    pub bindings: Vec<InxBinding>,
}

/// How this param's value combines with others that touch the same binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum InxMergeMode {
    /// Sums effects.
    Additive,
    /// Multiplies effects.
    Multiply,
    /// Overwrites (last wins).
    Override,
    /// Ignores existing values.
    Forced,
}

#[cfg(feature = "inx")]
impl From<inochi2d_parser::prelude::MergeMode> for InxMergeMode {
    fn from(mode: inochi2d_parser::prelude::MergeMode) -> Self {
        use inochi2d_parser::prelude::MergeMode::*;
        match mode {
            Additive => Self::Additive,
            Multiplicative => Self::Multiply,
            Override => Self::Override,
            Forced => Self::Forced,
        }
    }
}

/// A param mapped to a field of a specific node (`node_uuid`/`param_name`), with its
/// grid of interpolated values (`values`).
#[derive(Debug, Clone, Reflect)]
pub struct InxBinding {
    /// UUID of the target node.
    pub node_uuid: u32,
    /// Which field of the target node this binding drives.
    pub param_name: InxParamName,
    /// Interpolation between grid points.
    pub interpolation: InxInterpolation,
    /// Sampled values, shape depends on `param_name`.
    #[reflect(ignore)]
    pub values: InxBindingValues,
    /// Grid of explicitly authored points: [x][y] = true if authored (false = interpolated by the runtime).
    pub is_set: Vec<Vec<bool>>,
}

/// Raw sampled data of one `InxBinding`, in one of the shapes a param binding can carry.
#[derive(Debug, Clone, Default)]
pub enum InxBindingValues {
    /// Scalar values in a 2D grid (frames × values_per_frame) for TransformTX/TY/TZ/SX/SY/RX/RY/RZ and Opacity.
    Transform(InxFlatTransform),

    /// Vertex offsets in a 2D grid (frames × vertices_per_frame)
    Deform(InxFlatDeform),

    /// Fallback (not parsed)
    #[default]
    Other,
}

/// Flattened 2D grid of scalar values (one `InxParamName` per binding).
#[derive(Debug, Clone)]
pub struct InxFlatTransform {
    /// Flat buffer: data[frame * values_per_frame + y_idx]
    pub data: Vec<f32>,
    /// Points on the X axis (`axis_points[0].len()`).
    pub frames: usize,
    /// Points on the Y axis (`axis_points[1].len()`), or 1 if scalar.
    pub values_per_frame: usize,
}

impl InxFlatTransform {
    /// Value at `(frame, index)`, or `None` if out of buffer bounds.
    pub fn get(&self, frame: usize, index: usize) -> Option<f32> {
        let idx = frame * self.values_per_frame + index;
        self.data.get(idx).copied()
    }
}

/// Flattened 2D grid of per-vertex deform offsets.
#[derive(Debug, Clone)]
pub struct InxFlatDeform {
    /// Flat buffer: data[frame * vpf + vertex_idx] = [dx, dy]
    pub data: Vec<[f32; 2]>,
    /// Points on the X axis.
    pub frames: usize,
    /// [dx, dy] entries per X-axis point (Y-axis points × vertex count).
    pub vertices_per_frame: usize,
}

impl InxFlatDeform {
    /// Offset `[dx, dy]` at `(frame, vertex)`, or `None` if out of range.
    pub fn get(&self, frame: usize, vertex: usize) -> Option<[f32; 2]> {
        if frame >= self.frames || vertex >= self.vertices_per_frame {
            return None;
        }
        let idx = frame * self.vertices_per_frame + vertex;
        self.data.get(idx).copied()
    }
}

/// Node field that an [`InxBinding`] affects.
#[derive(Debug, Clone, Reflect)]
pub enum InxParamName {
    /// Translation X.
    TransformTX,
    /// Translation Y.
    TransformTY,
    /// Translation Z.
    TransformTZ,
    /// Scale X.
    TransformSX,
    /// Scale Y.
    TransformSY,
    /// Rotation X (radians).
    TransformRX,
    /// Rotation Y (radians).
    TransformRY,
    /// Rotation Z (radians, typically the one used).
    TransformRZ,
    /// Mesh deformation.
    Deform,
    /// Node opacity.
    Opacity,
    // Other(Box<str>),
    /// Other unrecognized field.
    Other,
}

/// Original node pose at load time (immutable). Used as the base on which param offsets are accumulated.
#[derive(Component, Debug, Clone, Reflect)]
pub struct InxBasePose {
    /// Base translation.
    pub translation: Vec3,
    /// Base rotation.
    pub rotation: Quat,
    /// Base scale.
    pub scale: Vec3,
}

/// Simple physics config (pendulum/spring-pendulum) parsed from a `SimplePhysics`
/// node, used by `simple_physics_system` to drive a param.
#[derive(Debug, Clone)]
pub struct SimplePhysicsConfig {
    /// UUID of the param this simulation drives.
    pub param_uuid: u32,
    /// Simulation type.
    pub model: simple_physics::PhysicsModel,
    /// How to map angle/length to the output parameter.
    pub map_mode: simple_physics::PhysicsMapMode,
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

/// Current value of each param for a puppet instance. Placed on the PuppetRoot entity.
#[derive(Component, Debug, Default, Reflect)]
pub struct InxParamState {
    /// param_uuid - current value [x, y] (y=0 if scalar)
    pub values: HashMap<u32, [f32; 2]>,
}

/// One param's resolved bindings: its handle + `(binding index, target entity)` for each
/// binding whose node exists in the spawned tree.
pub type ResolvedParam = (Handle<InxParam>, Vec<(u32, Entity)>);

/// Puppet's param bindings resolved to entities at spawn. Lives on the PuppetRoot entity, alongside [`InxParamState`].
///
/// The node uuid is a FILE concept: it references nodes within the same authored
/// puppet. It's resolved ONCE at spawn and the per-frame loop writes directly by
/// `Entity` - no per-frame uuid->entity maps and no possibility of resolving to
/// another puppet's node (or a runtime prop).
#[derive(Component, Debug, Default)]
pub struct InxResolvedBindings {
    /// For each param of the puppet: see [`ResolvedParam`].
    pub params: Vec<ResolvedParam>,
    /// Nodes with `InxDeform` (those with a mesh): the evaluator must write them the
    /// accumulated deform or zero them out every frame.
    pub deform_nodes: Vec<Entity>,
}

/// Multi-layer animation controller with crossfade. Replaces InxAnimationPlayer.
/// Placed on the PuppetRoot entity.
///
/// Usage:
/// ```ignore
/// // Play with a 0.3s crossfade
/// controller.play(anim_handle, 0.3);
///
/// // Play without crossfade (hard cut)
/// controller.play(anim_handle, 0.0);
///
/// // Perpetual idle layer (set once)
/// controller.set_idle(idle_handle);
///
/// // Play on a specific layer
/// controller.play_on_layer(1, anim_handle, true, 0.5);
/// ```
#[derive(Component, Debug, Default, Reflect)]
pub struct InxAnimationController {
    /// Animation layers, index 0 = base (idle), 1+ = actions. Evaluated in order: 0
    /// first (lowest priority), last = highest priority.
    pub layers: Vec<AnimationLayer>,

    /// Param defaults (used when no layer writes a param). Initialized with the
    /// puppet's defaults at spawn.
    #[reflect(ignore)]
    pub param_defaults: HashMap<u32, [f32; 2]>,

    /// If `true`, layer time/fade doesn't advance (frame frozen) but the params
    /// touched by the last evaluation are NOT reset to default - unlike
    /// `stop_all()`, which does reset. See [`InxAnimationController::pause`]/[`resume`](Self::resume).
    pub paused: bool,
}

/// An animation layer within [`InxAnimationController`] (crossfade, loop and speed are per layer).
#[derive(Debug, Clone, Reflect)]
pub struct AnimationLayer {
    /// Handle to the InxAnimation asset
    pub animation: Handle<InxAnimation>,
    /// Current time in seconds
    pub time: f32,
    /// Layer weight (0.0 = don't play, 1.0 = play at full strength)
    pub weight: f32,
    /// Playing?
    pub playing: bool,
    /// Looping?
    pub looping: bool,
    /// Speed (1.0 = normal)
    pub speed: f32,
    /// Transition state (fade in / fade out / no transition)
    pub fade: FadeState,
}

/// Weight transition state of an [`AnimationLayer`] during crossfade.
#[derive(Debug, Clone, Copy, Reflect)]
pub enum FadeState {
    /// No transition, stable weight.
    None,
    /// Fading in: weight rises from 0 -> 1.
    FadingIn {
        /// Total fade duration in seconds.
        duration: f32,
        /// Time elapsed since the fade started.
        elapsed: f32,
    },
    /// Fading out: weight drops from current -> 0 (removed on reaching 0).
    FadingOut {
        /// Total fade duration in seconds.
        duration: f32,
        /// Time elapsed since the fade started.
        elapsed: f32,
        /// Weight at the moment the fade started.
        start_weight: f32,
    },
}

impl InxAnimationController {
    /// Empty controller, no layers or defaults - use [`set_idle`](Self::set_idle)/
    /// [`play`](Self::play) to start playing.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            param_defaults: HashMap::default(),
            paused: false,
        }
    }

    /// Freezes the current frame: layer time/fade stops advancing, but already
    /// evaluated params do NOT revert to default (unlike [`stop_all`](Self::stop_all)). Reversible with [`resume`](Self::resume).
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resumes time/fade advancement after [`pause`](Self::pause).
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Plays an animation with crossfade. Fades out all active layers (except idle/layer 0)
    /// and fades in the new one on the action layer (layer 1).
    pub fn play(&mut self, animation: Handle<InxAnimation>, crossfade_secs: f32) -> usize {
        // Fade-out of every existing action layer (>= 1)
        for layer in self.layers.iter_mut().skip(1) {
            if layer.playing && layer.weight > 0.0 {
                layer.fade = FadeState::FadingOut {
                    duration: crossfade_secs.max(0.001),
                    elapsed: 0.0,
                    start_weight: layer.weight,
                };
            }
        }

        // New action layer
        let new_layer = AnimationLayer {
            animation,
            time: 0.0,
            weight: if crossfade_secs > 0.0 { 0.0 } else { 1.0 },
            playing: true,
            looping: false,
            speed: 1.0,
            fade: if crossfade_secs > 0.0 {
                FadeState::FadingIn {
                    duration: crossfade_secs,
                    elapsed: 0.0,
                }
            } else {
                FadeState::None
            },
        };

        self.layers.push(new_layer);
        self.layers.len() - 1
    }

    /// Plays an animation with crossfade and loop.
    pub fn play_looped(&mut self, animation: Handle<InxAnimation>, crossfade_secs: f32) {
        self.play(animation, crossfade_secs);
        if let Some(layer) = self.layers.last_mut() {
            layer.looping = true;
        }
    }

    /// Sets the idle animation (layer 0). Always loops, always active.
    pub fn set_idle(&mut self, animation: Handle<InxAnimation>) {
        let idle = AnimationLayer {
            animation,
            time: 0.0,
            weight: 1.0,
            playing: true,
            looping: true,
            speed: 1.0,
            fade: FadeState::None,
        };

        if self.layers.is_empty() {
            self.layers.push(idle);
        } else {
            self.layers[0] = idle;
        }
    }

    /// Plays on a specific layer.
    pub fn play_on_layer(
        &mut self,
        layer_idx: usize,
        animation: Handle<InxAnimation>,
        looping: bool,
        crossfade_secs: f32,
    ) {
        // Ensure enough layers exist
        while self.layers.len() <= layer_idx {
            self.layers.push(AnimationLayer {
                animation: Handle::default(),
                time: 0.0,
                weight: 0.0,
                playing: false,
                looping: false,
                speed: 1.0,
                fade: FadeState::None,
            });
        }

        let old = &mut self.layers[layer_idx];
        if old.playing && old.weight > 0.0 && crossfade_secs > 0.0 {
            // Fade out the old one, create the new one as an additional layer with
            // fade in
            old.fade = FadeState::FadingOut {
                duration: crossfade_secs,
                elapsed: 0.0,
                start_weight: old.weight,
            };

            self.layers.push(AnimationLayer {
                animation,
                time: 0.0,
                weight: 0.0,
                playing: true,
                looping,
                speed: 1.0,
                fade: FadeState::FadingIn {
                    duration: crossfade_secs,
                    elapsed: 0.0,
                },
            });
        } else {
            self.layers[layer_idx] = AnimationLayer {
                animation,
                time: 0.0,
                weight: 1.0,
                playing: true,
                looping,
                speed: 1.0,
                fade: FadeState::None,
            };
        }
    }

    /// Stops all action layers (not idle).
    pub fn stop_actions(&mut self, fade_out_secs: f32) {
        for layer in self.layers.iter_mut().skip(1) {
            if layer.playing && layer.weight > 0.0 {
                if fade_out_secs > 0.0 {
                    layer.fade = FadeState::FadingOut {
                        duration: fade_out_secs,
                        elapsed: 0.0,
                        start_weight: layer.weight,
                    };
                } else {
                    layer.playing = false;
                    layer.weight = 0.0;
                }
            }
        }
    }

    /// Stops everything (including idle).
    pub fn stop_all(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.playing = false;
            layer.weight = 0.0;
        }
    }
}

/// Animation loaded from the asset (`InxAnimationController` plays it by name, via `InxPuppet::named_animations`).
#[derive(Asset, Debug, Reflect)]
pub struct InxAnimation {
    /// Identifier name.
    pub name: String,
    /// Total duration in seconds.
    pub duration: f32,
    /// Duration of each frame in seconds.
    pub timestep: f32,
    /// Tracks controlling individual parameters.
    pub lanes: Vec<InxAnimationLane>,
}

/// Keyframe track of an animation, targeting a specific param.
#[derive(Debug, Clone, Reflect)]
pub struct InxAnimationLane {
    /// UUID of the target param.
    pub param_uuid: u32,
    /// Param component (0=X, 1=Y for vec2).
    pub target: u8,
    /// Interpolation between keyframes.
    pub interpolation: InxInterpolation,
    /// How this lane combines with other animations/base values.
    pub merge_mode: InxMergeMode,
    /// Keyframes ordered by frame.
    pub keyframes: Vec<InxKeyframe>,
}

/// Interpolation curve between a lane's keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum InxInterpolation {
    /// Linear interpolation.
    Linear,
    /// Jumps to the previous keyframe value (no smoothing).
    Stepped,
    /// Smooth cubic interpolation.
    Cubic,
}

#[cfg(feature = "inx")]
impl From<inochi2d_parser::prelude::Interpolation> for InxInterpolation {
    fn from(mode: inochi2d_parser::prelude::Interpolation) -> Self {
        use inochi2d_parser::prelude::Interpolation::*;
        match mode {
            Linear => Self::Linear,
            Stepped | Nearest => Self::Stepped,
            Cubic => Self::Cubic,
        }
    }
}

/// A point of an [`InxAnimationLane`] (`tension` is only used with `Cubic`).
#[derive(Debug, Clone, Copy, Reflect)]
pub struct InxKeyframe {
    /// Frame index.
    pub frame: u32,
    /// Value at this frame.
    pub value: f32,
    /// Tension for cubic interpolation (0.0-1.0).
    pub tension: f32,
}

/// Marker component for entities spawned from a puppet.
#[derive(Component, Clone, Reflect)]
pub struct InxPuppetRoot {
    /// The asset this instance was spawned from.
    pub source: Handle<InxPuppet>,
}
