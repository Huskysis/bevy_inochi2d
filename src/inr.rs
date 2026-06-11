//! Standalone reader for the INR runtime format (v1).
//!
//! INR is a GLB-style container: `b"INR1"` magic + u32 version + a JSON
//! index chunk + one binary blob. Everything cross-references by array
//! index (never UUID) and the node list is flattened in pre-order, which
//! maps directly onto ECS spawning. Textures are raw RGBA8 (premultiplied,
//! sRGB-encoded) ready for upload without image decoding.
//!
//! This module is self-contained: it only needs `serde_json` + `bytemuck`.

use serde::Deserialize;
use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"INR1";
pub const VERSION: u32 = 1;

#[derive(Debug, Error)]
pub enum InrError {
    #[error("not an INR file (bad magic)")]
    BadMagic,
    #[error("unsupported INR container version {0}")]
    UnsupportedVersion(u32),
    #[error("truncated INR container")]
    Truncated,
    #[error("invalid JSON chunk: {0}")]
    Json(#[from] serde_json::Error),
    #[error("buffer view {0} out of range")]
    BadView(u32),
    #[error("texture {0}: unsupported pixel format")]
    UnsupportedTexture(usize),
}

// --- string enums (unknown values fall back to spec defaults) --------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrNodeKind {
    Part,
    Composite,
    Mask,
    #[serde(rename = "meshgroup")]
    MeshGroup,
    #[serde(rename = "simplephysics")]
    SimplePhysics,
    Camera,
    #[default]
    #[serde(other)]
    Node,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrBlendMode {
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
    #[default]
    #[serde(other)]
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMaskMode {
    Dodge,
    #[default]
    #[serde(other)]
    Mask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrPhysicsModel {
    SpringPendulum,
    #[default]
    #[serde(other)]
    Pendulum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMapMode {
    Xy,
    LengthAngle,
    Yx,
    #[default]
    #[serde(other)]
    AngleLength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrMergeMode {
    Multiplicative,
    Override,
    Forced,
    #[default]
    #[serde(other)]
    Additive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrInterpolation {
    Stepped,
    Nearest,
    Cubic,
    #[default]
    #[serde(other)]
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrBindingKind {
    #[default]
    Scalar,
    Deform,
    /// Unknown kinds must not be misread as scalar data.
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrBindingTarget {
    #[serde(rename = "transform.t.x")]
    TranslateX,
    #[serde(rename = "transform.t.y")]
    TranslateY,
    #[serde(rename = "transform.t.z")]
    TranslateZ,
    #[serde(rename = "transform.r.x")]
    RotateX,
    #[serde(rename = "transform.r.y")]
    RotateY,
    #[serde(rename = "transform.r.z")]
    RotateZ,
    #[serde(rename = "transform.s.x")]
    ScaleX,
    #[serde(rename = "transform.s.y")]
    ScaleY,
    Deform,
    Opacity,
    #[default]
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrTextureFormat {
    #[default]
    Rgba8,
    Bc7,
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InrColorSpace {
    Linear,
    #[default]
    #[serde(other)]
    Srgb,
}

/// Parsed container: JSON document + binary blob.
pub struct InrModel {
    pub doc: InrDoc,
    pub bin: Vec<u8>,
}

impl InrModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, InrError> {
        if bytes.len() < 16 {
            return Err(InrError::Truncated);
        }
        if bytes[0..4] != MAGIC {
            return Err(InrError::BadMagic);
        }
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let version = u32_at(4);
        if version != VERSION {
            return Err(InrError::UnsupportedVersion(version));
        }
        let json_len = u32_at(8) as usize;
        let bin_len = u32_at(12) as usize;
        let json_end = 16usize.checked_add(json_len).ok_or(InrError::Truncated)?;
        let bin_end = json_end.checked_add(bin_len).ok_or(InrError::Truncated)?;
        if bytes.len() < bin_end {
            return Err(InrError::Truncated);
        }
        let doc: InrDoc = serde_json::from_slice(&bytes[16..json_end])?;
        Ok(Self {
            doc,
            bin: bytes[json_end..bin_end].to_vec(),
        })
    }

    pub fn view_bytes(&self, view: u32) -> Result<&[u8], InrError> {
        let v = self
            .doc
            .buffer_views
            .get(view as usize)
            .ok_or(InrError::BadView(view))?;
        let start = v.offset as usize;
        let end = start
            .checked_add(v.length as usize)
            .ok_or(InrError::BadView(view))?;
        self.bin.get(start..end).ok_or(InrError::BadView(view))
    }

    /// Copying read: safe regardless of `bin` alignment.
    pub fn view_f32(&self, view: u32) -> Result<Vec<f32>, InrError> {
        let b = self.view_bytes(view)?;
        if b.len() % 4 != 0 {
            return Err(InrError::BadView(view));
        }
        Ok(bytemuck::pod_collect_to_vec(b))
    }

    pub fn view_u32(&self, view: u32) -> Result<Vec<u32>, InrError> {
        let b = self.view_bytes(view)?;
        if b.len() % 4 != 0 {
            return Err(InrError::BadView(view));
        }
        Ok(bytemuck::pod_collect_to_vec(b))
    }
}

// --- JSON schema (unknown fields are ignored for forward compat) ----------

#[derive(Debug, Deserialize)]
pub struct InrDoc {
    #[serde(default)]
    pub meta: Meta,
    pub physics: Physics,
    pub buffer_views: Vec<BufferView>,
    #[serde(default)]
    pub textures: Vec<TextureDesc>,
    /// Flattened pre-order: a parent always precedes its children.
    pub nodes: Vec<InrNode>,
    #[serde(default)]
    pub params: Vec<InrParam>,
    #[serde(default)]
    pub animations: Vec<InrAnimation>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Meta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub rigger: Option<String>,
    #[serde(default)]
    pub artist: Option<String>,
    #[serde(default)]
    pub rights: Option<String>,
    #[serde(default)]
    pub copyright: Option<String>,
    #[serde(default)]
    pub license_url: Option<String>,
    #[serde(default)]
    pub contact: Option<String>,
    #[serde(default)]
    pub reference: Option<String>,
    #[serde(default)]
    pub source_version: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Physics {
    pub pixels_per_meter: f32,
    pub gravity: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct BufferView {
    pub offset: u32,
    pub length: u32,
}

#[derive(Debug, Deserialize)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    /// Pixel layout, currently always `Rgba8`.
    pub format: InrTextureFormat,
    /// Encoding of the RGB channels.
    #[serde(default)]
    pub color_space: InrColorSpace,
    /// RGB premultiplied by alpha (in `color_space`).
    #[serde(default)]
    pub premultiplied: bool,
    pub view: u32,
}

#[derive(Debug, Deserialize)]
pub struct InrNode {
    pub name: String,
    pub uuid: u32,
    /// Index into `nodes`; absent on the root.
    #[serde(default)]
    pub parent: Option<u32>,
    pub kind: InrNodeKind,
    pub enabled: bool,
    pub zsort: f32,
    #[serde(default)]
    pub lock_to_root: bool,
    pub translation: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 2],
    #[serde(default)]
    pub mesh: Option<InrMesh>,
    #[serde(default)]
    pub part: Option<InrPart>,
    #[serde(default)]
    pub composite: Option<InrComposite>,
    #[serde(default)]
    pub physics: Option<InrPhysics>,
}

#[derive(Debug, Deserialize)]
pub struct InrMesh {
    pub vertex_count: u32,
    /// View: f32 x/y pairs (`vertex_count * 2` floats).
    pub positions: u32,
    /// View: f32 u/v pairs.
    pub uvs: u32,
    /// View: u32 triangle indices.
    pub indices: u32,
    pub origin: [f32; 2],
}

#[derive(Debug, Deserialize)]
pub struct InrPart {
    /// Texture indices [albedo, emissive, bump]; -1 = none.
    pub textures: [i32; 3],
    pub blend_mode: InrBlendMode,
    pub tint: [f32; 3],
    pub screen_tint: [f32; 3],
    pub opacity: f32,
    #[serde(default)]
    pub emission_strength: f32,
    pub mask_threshold: f32,
    #[serde(default)]
    pub masks: Vec<InrMask>,
}

#[derive(Debug, Deserialize)]
pub struct InrComposite {
    pub blend_mode: InrBlendMode,
    pub tint: [f32; 3],
    pub screen_tint: [f32; 3],
    pub opacity: f32,
    pub mask_threshold: f32,
    #[serde(default)]
    pub masks: Vec<InrMask>,
}

#[derive(Debug, Deserialize)]
pub struct InrMask {
    /// Index into `nodes`.
    pub node: u32,
    pub mode: InrMaskMode,
}

#[derive(Debug, Deserialize)]
pub struct InrPhysics {
    /// Index into `params`; -1 = unresolved.
    pub param: i32,
    pub model: InrPhysicsModel,
    pub map_mode: InrMapMode,
    pub gravity: f32,
    pub length: f32,
    pub frequency: f32,
    pub angle_damping: f32,
    pub length_damping: f32,
    pub output_scale: [f32; 2],
    #[serde(default)]
    pub local_only: bool,
}

#[derive(Debug, Deserialize)]
pub struct InrParam {
    pub name: String,
    pub uuid: u32,
    pub is_vec2: bool,
    pub min: [f32; 2],
    pub max: [f32; 2],
    pub defaults: [f32; 2],
    pub axis_points: [Vec<f32>; 2],
    pub merge_mode: InrMergeMode,
    #[serde(default)]
    pub bindings: Vec<InrBinding>,
}

#[derive(Debug, Deserialize)]
pub struct InrBinding {
    /// Index into `nodes`.
    pub node: u32,
    pub target: InrBindingTarget,
    pub interpolation: InrInterpolation,
    pub x_count: u32,
    pub y_count: u32,
    /// Row-major [x][y] authored flags, flattened.
    pub is_set: Vec<bool>,
    pub kind: InrBindingKind,
    pub view: u32,
}

#[derive(Debug, Deserialize)]
pub struct InrAnimation {
    pub name: String,
    pub timestep: f32,
    #[serde(default)]
    pub additive: bool,
    pub length: u32,
    #[serde(default)]
    pub lead_in: u32,
    #[serde(default)]
    pub lead_out: u32,
    #[serde(default = "one")]
    pub weight: f32,
    #[serde(default)]
    pub lanes: Vec<InrLane>,
}

fn one() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct InrLane {
    /// Index into `params`; -1 = unresolved.
    pub param: i32,
    /// 0 = X, 1 = Y.
    pub target: u8,
    pub interpolation: InrInterpolation,
    pub merge_mode: InrMergeMode,
    /// [frame, value, tension] per keyframe.
    pub keyframes: Vec<[f32; 3]>,
}
