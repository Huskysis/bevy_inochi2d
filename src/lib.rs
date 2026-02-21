pub mod animation;
pub mod auto_spawn;
pub mod grid_interpolation;
pub mod loader;
pub mod pipeline;
pub mod plugin;
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
use inochi2d_parser::owned::Puppet;

pub mod prelude {
    pub use crate::{
        BlendMode, InxAnimationController, InxMaterial, InxNode, InxNodeType, InxParam, InxPuppet,
        InxPuppetRoot, InxScene, InxUUID, MeshWrap,
        plugin::{Inochi2dPlugin, InxAnimationPlugin},
    };
    pub use inochi2d_parser::prelude::{MaskMode, Transform as InxTransform};
}

/// Componente-comando para spawnear un puppet como escena.
///
/// Uso:
/// ```ignore
///
/// let my_puppet_handle: Handle<InxPuppet> = ...;
///
/// commands.spawn(InxScene {
///     puppet: my_puppet_handle,
///     transform: Transform::from_xyz(0.0, 0.0, 0.0),
///     animation: true,
/// });
/// ```
#[derive(Component)]
pub struct InxScene {
    /// Handle al asset InxPuppet cargado.
    pub puppet: Handle<InxPuppet>,
    /// Transform raiz de la instancia.
    pub transform: Transform,
    /// Si true, inserta InxAnimationController con todas las animaciones.
    pub animation: bool,
}

#[derive(Debug, Asset, TypePath)]
pub struct InxPuppet {
    pub nodes: Vec<InxNode>,
    pub params: Vec<Handle<InxParam>>,
    pub animations: Vec<Handle<InxAnimation>>,
    pub textures: Vec<Handle<Image>>,

    pub named_nodes: HashMap<Box<str>, InxNode>,
    // pub named_mesh: HashMap<Box<str>, Handle<Mesh>>,
    // pub named_material: HashMap<Box<str>, Handle<Material2d>>,
    pub named_params: HashMap<Box<str>, Handle<InxParam>>,
    pub named_animations: HashMap<Box<str>, Handle<InxAnimation>>,

    // Metadata
    pub meta: InxMeta,
    pub physics: InxPhysics,

    pub source: Option<Puppet>,
}

#[derive(Debug, Clone)]
pub struct InxMeta {
    pub name: String,
    pub version: String,
    pub rigger: String,
    pub artist: String,
    pub rights: String,
    pub copyright: String,
    pub license_url: String,
    pub contact: String,
    pub reference: String,
    pub thumbnail_id: u32,
    pub preserve_pixels: bool,
}

#[derive(Debug, Clone, Component)]
pub struct InxPhysics {
    pub pixels_per_meter: f32,
    pub gravity: f32,
}

/// Nodo del árbol del puppet.
#[derive(Asset, Debug, TypePath, Component, Clone)]
pub struct InxNode {
    pub uuid: u32,
    pub name: Box<str>,
    pub node_type: InxNodeType,
    pub material: Option<InxMaterial>,
    pub transform: Transform,
    pub zsort: f32,
    pub enabled: bool,
    pub physics_data: Option<SimplePhysicsConfig>,
    pub children: Vec<InxNode>,
}

// Simplificado por ahora
#[derive(Debug, Clone, Copy, Component, PartialEq, Eq, Reflect)]
pub enum InxNodeType {
    Part,
    Composite,
    Mask,
    MeshGroup,
    Camera,
    SimplePhysics,
    Generic,
}

#[derive(Default, Clone, Copy, Component, ExtractComponent, Debug, Reflect)]
#[require(VisibilityClass)]
#[component(on_add = add_visibility_class::<InxUUID>)]
pub struct InxUUID(pub u32);

#[derive(Clone, Default, Component, Reflect)]
pub struct MeshWrap(pub Handle<Mesh>);

#[derive(Clone, Default, Component, Reflect)]
pub struct InxMeshWrap(pub Handle<InxMesh>);

#[derive(Asset, Debug, Component, Default, Reflect)]
pub struct InxMesh {
    pub vertex_buffer: Vec<[f32; 2]>,
    pub uv_buffer: Vec<[f32; 2]>,
    pub index_buffer: Vec<u32>,
    pub origin: Vec2, // offset de la textura NO DE TRANSFORM
}

/// Material 2D para Inochi2d.
#[derive(Asset, Debug, Clone, Component, Default, Reflect)]
#[require(InxUUID, InxZSort, Transform, Visibility)]
pub struct InxMaterial {
    #[reflect(ignore)]
    pub mesh: Option<Arc<InxMesh>>,
    pub texture_albedo: Option<Handle<Image>>,
    pub texture_emissive: Option<Handle<Image>>,
    pub texture_bumpmap: Option<Handle<Image>>,
    pub textures: [u32; 3],
    pub tint: Vec3,
    pub screen_tint: Vec3,
    pub opacity: f32,
    pub emissive_strength: f32,
    pub mask_threshold: f32,
    pub blend_mode: BlendMode,
    pub masks: Vec<InxMask>,
}

#[derive(Debug, Clone, Copy, Default, Component, PartialEq, Reflect)]
pub struct InxZSort(pub f32);

// Sera utilizado el del parser, no este, no es necesario duplicar lo mismo
// Por ahora, se utilizara para debug
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, Reflect)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    LinearDodge,
    Add,
    ColorBurn,
    HardLight,
    SoftLight,
    Subtract,
    Difference,
    Exclusion,
    Inverse,
    DestinationIn,
    ClipToLower,
    SliceFromLower,
}

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
    fn blend_state(&self) -> BlendState {
        use bevy::render::render_resource::{BlendFactor::*, BlendOperation::*};
        match self {
            // Normal: src * srcα + dst * (1 - srcα)
            BlendMode::Normal =>
            // BlendState::REPLACE,
            // BlendState::PREMULTIPLIED_ALPHA_BLENDING,
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

            // Overlay: src * dstα + dst * (1 - srcα)
            // Aprox. (Photoshot-like)
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

            // DestinationOut: dstα - src
            // BlendMode::DestinationOut => BlendState {
            //     color: BlendComponent {
            //         src_factor: DstAlpha,
            //         dst_factor: Zero,
            //         operation: Add,
            //     },
            //     alpha: BlendComponent {
            //         src_factor: DstAlpha,
            //         dst_factor: Zero,
            //         operation: Add,
            //     },
            // },

            // ClipToLower: src * dstα + dst * (1 - srcα)
            // Clips source to destination alpha (shows only where dst exists)
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

            // SliceFromLower: dst * (1 - srcα) - src * dstα
            // Cuts out source from destination
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

            // SourceIn: srcα - dst
            // BlendMode::SourceIn => BlendState {
            //     color: BlendComponent {
            //         src_factor: DstAlpha,
            //         dst_factor: Zero,
            //         operation: Add,
            //     },
            //     alpha: BlendComponent {
            //         src_factor: DstAlpha,
            //         dst_factor: Zero,
            //         operation: Add,
            //     },
            // },

            // SourceOut: src - dstα
            // BlendMode::SourceOut => BlendState {
            //     color: BlendComponent {
            //         src_factor: OneMinusDstAlpha,
            //         dst_factor: Zero,
            //         operation: Zero,
            //     },
            //     alpha: BlendComponent {
            //         src_factor: OneMinusDstAlpha,
            //         dst_factor: Zero,
            //         operation: Add,
            //     },
            // },

            // Complex modes - fallback to Normal (need multi-pass for accurate)
            // TODO: Implement via shader or multi-pass
            // BlendMode::ColorBurn
            // | BlendMode::HardLight
            // | BlendMode::SoftLight
            // | BlendMode::Difference
            // | BlendMode::Exclusion
            // | BlendMode::Inverse
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Reflect)]
pub struct InxMask {
    pub source_uuid: u32,
    pub mode: InxMaskMode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Reflect)]
pub enum InxMaskMode {
    #[default]
    Mask,
    Dodge,
}

impl From<inochi2d_parser::owned::MaskMode> for InxMaskMode {
    fn from(mode: inochi2d_parser::owned::MaskMode) -> Self {
        use inochi2d_parser::owned::MaskMode::*;
        match mode {
            Mask => Self::Mask,
            Dodge => Self::Dodge,
        }
    }
}

/// Offsets de deformacion por vertice, actualizados cada frame.
/// Se acumulan desde multiples params y se aplican en el shader.
#[derive(Component, Debug, Clone, Default, Reflect)]
pub struct InxDeform {
    /// Offset [dx, dy] por vertice. Longitud = vertex count del mesh.
    pub offsets: Vec<[f32; 2]>,
}

// Actualizar InxParam (añadir axis_points):
#[derive(Asset, Debug, Component, Reflect)]
pub struct InxParam {
    pub uuid: u32,
    pub name: String,
    pub is_vec2: bool,
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub defaults: [f32; 2],
    pub axis_points: [Vec<f32>; 2],
    pub merge_mode: InxMergeMode,
    pub bindings: Vec<InxBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum InxMergeMode {
    Additive,
    Multiply,
    Override,
    Forced,
}

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

// Actualizar InxBinding:
#[derive(Debug, Clone, Reflect)]
pub struct InxBinding {
    pub node_uuid: u32,
    pub param_name: InxParamName,
    pub interpolation: InxInterpolation,
    #[reflect(ignore)]
    pub values: InxBindingValues,
    pub is_set: Vec<Vec<bool>>,
}

#[derive(Debug, Clone, Default)]
pub enum InxBindingValues {
    /// Valores escalares en grilla 2D (frames × values_per_frame)
    /// Para TransformTX/TY/TZ/SX/SY/RX/RY/RZ y Opacity
    Transform(InxFlatTransform),

    /// Offsets de vertices en grilla 2D (frames × vertices_per_frame)
    Deform(InxFlatDeform),

    /// Fallback (no parseado)
    #[default]
    Other,
}

#[derive(Debug, Clone)]
pub struct InxFlatTransform {
    /// Buffer plano: data[frame * values_per_frame + y_idx]
    pub data: Vec<f32>,
    pub frames: usize,           // axis_points[0].len() (X axis)
    pub values_per_frame: usize, // axis_points[1].len() (Y axis) - o 1 si scalar
}

impl InxFlatTransform {
    pub fn get(&self, frame: usize, index: usize) -> Option<f32> {
        let idx = frame * self.values_per_frame + index;
        self.data.get(idx).copied()
    }
}

#[derive(Debug, Clone)]
pub struct InxFlatDeform {
    /// Buffer plano: data[frame * vpf + vertex_idx] = [dx, dy]
    pub data: Vec<[f32; 2]>,
    pub frames: usize,
    pub vertices_per_frame: usize,
}

impl InxFlatDeform {
    pub fn get(&self, frame: usize, vertex: usize) -> Option<[f32; 2]> {
        if frame >= self.frames || vertex >= self.vertices_per_frame {
            return None;
        }
        let idx = frame * self.vertices_per_frame + vertex;
        self.data.get(idx).copied()
    }
}

#[derive(Debug, Clone, Reflect)]
pub enum InxParamName {
    TransformTX,
    TransformTY,
    TransformTZ,
    TransformSX,
    TransformSY,
    TransformRX,
    TransformRY,
    TransformRZ,
    Deform,
    Opacity,
    // Other(Box<str>),
    Other,
}

/// Pose original del nodo al cargarse (inmutable).
/// Se usa como base sobre la que se acumulan offsets de params.
#[derive(Component, Debug, Clone, Reflect)]
pub struct InxBasePose {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub opacity: f32,
}

#[derive(Debug, Clone)]
pub struct SimplePhysicsConfig {
    pub param_uuid: u32,
    pub model: simple_physics::PhysicsModel,
    pub map_mode: simple_physics::PhysicsMapMode,
    pub gravity: f32,
    pub length: f32,
    pub frequency: f32,
    pub angle_damping: f32,
    pub length_damping: f32,
    pub output_scale: [f32; 2],
    pub local_only: bool,
}

/// Valor actual de cada param para una instancia de puppet.
/// Se coloca en la entidad PuppetRoot.
#[derive(Component, Debug, Default, Reflect)]
pub struct InxParamState {
    /// param_uuid - valor actual [x, y] (y=0 si escalar)
    pub values: HashMap<u32, [f32; 2]>,
}

/// Controlador de animaciones multi-capa con crossfade.
/// Reemplaza InxAnimationPlayer. Se coloca en la entidad PuppetRoot.
///
/// Uso:
/// ```ignore
/// // Reproducir con crossfade de 0.3s
/// controller.play(anim_handle, 0.3);
///
/// // Reproducir sin crossfade (corte directo)
/// controller.play(anim_handle, 0.0);
///
/// // Capa idle perpetua (se setea una vez)
/// controller.set_idle(idle_handle);
///
/// // Reproducir en capa específica
/// controller.play_on_layer(1, anim_handle, true, 0.5);
/// ```
#[derive(Component, Debug, Default, Reflect)]
pub struct InxAnimationController {
    /// Capas de animación, índice 0 = base (idle), 1+ = acciones.
    /// Se evalúan en orden: 0 primero (menor prioridad), último = mayor prioridad.
    pub layers: Vec<AnimationLayer>,

    /// Defaults de params (se usan cuando ninguna capa escribe un param).
    /// Se inicializa con los defaults del puppet al spawnear.
    #[reflect(ignore)]
    pub param_defaults: HashMap<u32, [f32; 2]>,
}

#[derive(Debug, Clone, Reflect)]
pub struct AnimationLayer {
    /// Handle al InxAnimation asset
    pub animation: Handle<InxAnimation>,
    /// Tiempo actual en segundos
    pub time: f32,
    /// Peso de la capa (0.0 = no reproducir, 1.0 = reproducir completamente)
    pub weight: f32,
    /// Reproduciendo?
    pub playing: bool,
    /// En bucle?
    pub looping: bool,
    /// Velocidad (1.0 = normal)
    pub speed: f32,
    /// Estado de transicion(Entrada/Salida/Sin transicion)
    pub fade: FadeState,
}

#[derive(Debug, Clone, Copy, Reflect)]
pub enum FadeState {
    /// Sin transición, peso estable.
    None,
    /// Entrando: peso sube de 0 → 1.
    FadingIn { duration: f32, elapsed: f32 },
    /// Saliendo: peso baja de current → 0 (se elimina al llegar a 0).
    FadingOut {
        duration: f32,
        elapsed: f32,
        start_weight: f32,
    },
}

impl InxAnimationController {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            param_defaults: HashMap::default(),
        }
    }

    /// Reproduce una animación con crossfade.
    /// Hace fade-out de todas las capas activas (excepto idle/layer 0)
    /// y fade-in de la nueva en la capa de acción (layer 1).
    pub fn play(&mut self, animation: Handle<InxAnimation>, crossfade_secs: f32) -> usize {
        // Fade-out de todas las capas de acción existentes (>= 1)
        for layer in self.layers.iter_mut().skip(1) {
            if layer.playing && layer.weight > 0.0 {
                layer.fade = FadeState::FadingOut {
                    duration: crossfade_secs.max(0.001),
                    elapsed: 0.0,
                    start_weight: layer.weight,
                };
            }
        }

        // Nueva capa de acción
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

    /// Reproduce una animación con crossfade y loop.
    pub fn play_looped(&mut self, animation: Handle<InxAnimation>, crossfade_secs: f32) {
        self.play(animation, crossfade_secs);
        if let Some(layer) = self.layers.last_mut() {
            layer.looping = true;
        }
    }

    /// Setea la animación idle (layer 0). Siempre loop, siempre activa.
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

    /// Reproduce en una capa específica.
    pub fn play_on_layer(
        &mut self,
        layer_idx: usize,
        animation: Handle<InxAnimation>,
        looping: bool,
        crossfade_secs: f32,
    ) {
        // Asegurar que existen suficientes capas
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
            // Fade out del viejo, crear nuevo como layer adicional con fade in
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

    /// Detiene todas las capas de acción (no el idle).
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

    /// Detiene todo (incluyendo idle).
    pub fn stop_all(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.playing = false;
            layer.weight = 0.0;
        }
    }
}

#[derive(Asset, Debug, Reflect)]
pub struct InxAnimation {
    pub name: String,
    pub duration: f32,
    pub timestep: f32,
    pub lanes: Vec<InxAnimationLane>,
}

#[derive(Debug, Clone, Reflect)]
pub struct InxAnimationLane {
    pub param_uuid: u32,
    pub target: u8,
    pub interpolation: InxInterpolation,
    pub merge_mode: InxMergeMode,
    pub keyframes: Vec<InxKeyframe>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Reflect)]
pub enum InxInterpolation {
    Linear,
    Stepped,
    Cubic,
}

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

#[derive(Debug, Clone, Copy, Reflect)]
pub struct InxKeyframe {
    pub frame: u32,
    pub value: f32,
    pub tension: f32,
}

/// Componente marcador para entidades spawneadas desde un puppet.
#[derive(Component, Clone, Reflect)]
pub struct InxPuppetRoot {
    pub source: Handle<InxPuppet>,
}
