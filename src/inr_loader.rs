//! Asset loader for `.inr` files.
//!
//! INR's flat pre-order node list, index-based cross-references and raw
//! RGBA8 textures let this loader build assets without image decoding,
//! UUID maps or nested-JSON parsing.

use std::sync::Arc;

use bevy::{
    asset::{AssetLoader, LoadContext, RenderAssetUsages},
    image::{
        Image, ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor,
    },
    platform::collections::HashMap,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use crate::{
    BlendMode, InxAnimation, InxAnimationLane, InxBinding, InxBindingValues, InxFlatDeform,
    InxFlatTransform, InxInterpolation, InxKeyframe, InxMask, InxMaskMode, InxMaterial,
    InxMergeMode, InxMesh, InxMeta, InxNode, InxNodeType, InxParam, InxParamName, InxPhysics,
    InxPuppet, SimplePhysicsConfig,
    inr::{self, InrModel},
    simple_physics::{PhysicsMapMode, PhysicsModel},
};

#[derive(TypePath)]
pub struct InrLoader;

impl AssetLoader for InrLoader {
    type Asset = InxPuppet;
    type Settings = ();
    type Error = inr::InrError;

    async fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| inr::InrError::Truncated)?;
        let model = InrModel::from_bytes(&bytes)?;
        convert(&model, load_context)
    }

    fn extensions(&self) -> &[&str] {
        &["inr"]
    }
}

fn convert(model: &InrModel, ctx: &mut LoadContext<'_>) -> Result<InxPuppet, inr::InrError> {
    let doc = &model.doc;

    // Textures use sRGB views: hardware decodes to linear on sample, blending
    // happens in linear space and the shader premultiplies after sampling.
    // Data must be straight alpha; legacy INR files with premultiplied
    // textures are un-premultiplied here (in gamma space, matching how they
    // were premultiplied).
    let mut textures = Vec::with_capacity(doc.textures.len());
    for (i, t) in doc.textures.iter().enumerate() {
        if t.format != inr::InrTextureFormat::Rgba8 {
            return Err(inr::InrError::UnsupportedTexture(i));
        }
        let mut data = model.view_bytes(t.view)?.to_vec();
        if t.premultiplied {
            for px in data.chunks_exact_mut(4) {
                let a = px[3] as u32;
                if a > 0 && a < 255 {
                    for c in &mut px[..3] {
                        *c = ((*c as u32 * 255 + a / 2) / a).min(255) as u8;
                    }
                }
            }
        }
        let mut image = Image::new(
            Extent3d {
                width: t.width,
                height: t.height,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            label: Some(format!("texture_{i}")),
            address_mode_u: ImageAddressMode::ClampToBorder,
            address_mode_v: ImageAddressMode::ClampToBorder,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            ..Default::default()
        });
        textures.push(ctx.add_labeled_asset(format!("texture_{i}"), image));
    }

    // Children lists (doc order is pre-order, parents precede children).
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); doc.nodes.len()];
    let mut root = None;
    for (i, n) in doc.nodes.iter().enumerate() {
        match n.parent {
            Some(p) => children[p as usize].push(i),
            None => root = Some(i),
        }
    }
    let root = root.unwrap_or(0);

    let mut nodes = Vec::with_capacity(doc.nodes.len());
    let mut named_nodes = HashMap::default();
    build_node(model, root, &children, &textures, &mut nodes, &mut named_nodes)?;

    // Params (bindings reference nodes by index; resolve to uuid here).
    let mut params = Vec::with_capacity(doc.params.len());
    let mut named_params = HashMap::default();
    for p in &doc.params {
        let asset = convert_param(model, p)?;
        let handle = ctx.add_labeled_asset(format!("param_{}", p.uuid), asset);
        named_params.insert(p.name.clone().into_boxed_str(), handle.clone());
        params.push(handle);
    }

    let mut animations = Vec::with_capacity(doc.animations.len());
    let mut named_animations = HashMap::default();
    for a in &doc.animations {
        let asset = convert_animation(doc, a);
        let handle = ctx.add_labeled_asset(format!("anim_{}", a.name), asset);
        named_animations.insert(a.name.clone().into_boxed_str(), handle.clone());
        animations.push(handle);
    }

    let m = &doc.meta;
    let meta = InxMeta {
        name: m.name.clone().unwrap_or_default(),
        version: m.source_version.clone().unwrap_or_default(),
        rigger: m.rigger.clone().unwrap_or_default(),
        artist: m.artist.clone().unwrap_or_default(),
        rights: m.rights.clone().unwrap_or_default(),
        copyright: m.copyright.clone().unwrap_or_default(),
        license_url: m.license_url.clone().unwrap_or_default(),
        contact: m.contact.clone().unwrap_or_default(),
        reference: m.reference.clone().unwrap_or_default(),
        thumbnail_id: 0,
        preserve_pixels: false,
    };

    Ok(InxPuppet {
        nodes,
        params,
        animations,
        textures,
        named_nodes,
        named_params,
        named_animations,
        meta,
        physics: InxPhysics {
            pixels_per_meter: doc.physics.pixels_per_meter,
            gravity: doc.physics.gravity,
        },
    })
}

/// Post-order build (children first), pushing each node into `nodes` so the
/// root ends up last — same contract the spawner expects.
fn build_node(
    model: &InrModel,
    index: usize,
    children: &[Vec<usize>],
    images: &[Handle<Image>],
    nodes: &mut Vec<InxNode>,
    named_nodes: &mut HashMap<Box<str>, InxNode>,
) -> Result<InxNode, inr::InrError> {
    let doc = &model.doc;
    let n = &doc.nodes[index];

    let mut child_nodes = Vec::with_capacity(children[index].len());
    for &c in &children[index] {
        child_nodes.push(build_node(model, c, children, images, nodes, named_nodes)?);
    }

    let node_type = match n.kind {
        inr::InrNodeKind::Part => InxNodeType::Part,
        inr::InrNodeKind::Composite => InxNodeType::Composite,
        inr::InrNodeKind::Mask => InxNodeType::Mask,
        inr::InrNodeKind::MeshGroup => InxNodeType::MeshGroup,
        inr::InrNodeKind::Camera => InxNodeType::Camera,
        inr::InrNodeKind::SimplePhysics => InxNodeType::SimplePhysics,
        inr::InrNodeKind::Node => InxNodeType::Generic,
    };

    let mesh = n.mesh.as_ref().map(|m| convert_mesh(model, m)).transpose()?;

    let material = if let Some(part) = &n.part {
        Some(InxMaterial {
            mesh: mesh.map(Arc::new),
            texture_albedo: tex_handle(images, part.textures[0]),
            texture_emissive: tex_handle(images, part.textures[1]),
            texture_bumpmap: tex_handle(images, part.textures[2]),
            textures: part.textures.map(|t| if t < 0 { u32::MAX } else { t as u32 }),
            tint: part.tint.into(),
            screen_tint: part.screen_tint.into(),
            opacity: part.opacity,
            emissive_strength: part.emission_strength,
            mask_threshold: part.mask_threshold,
            blend_mode: part.blend_mode.into(),
            masks: convert_masks(doc, &part.masks),
        })
    } else if let Some(comp) = &n.composite {
        Some(InxMaterial {
            mesh: None,
            texture_albedo: None,
            texture_emissive: None,
            texture_bumpmap: None,
            textures: [u32::MAX; 3],
            tint: comp.tint.into(),
            screen_tint: comp.screen_tint.into(),
            opacity: comp.opacity,
            emissive_strength: 0.0,
            mask_threshold: comp.mask_threshold,
            blend_mode: comp.blend_mode.into(),
            masks: convert_masks(doc, &comp.masks),
        })
    } else {
        None
    };

    let physics_data = n.physics.as_ref().map(|p| SimplePhysicsConfig {
        param_uuid: p
            .param
            .try_into()
            .ok()
            .and_then(|i: usize| doc.params.get(i))
            .map_or(u32::MAX, |param| param.uuid),
        model: match p.model {
            inr::InrPhysicsModel::SpringPendulum => PhysicsModel::SpringPendulum,
            inr::InrPhysicsModel::Pendulum => PhysicsModel::Pendulum,
        },
        map_mode: match p.map_mode {
            inr::InrMapMode::Xy => PhysicsMapMode::XY,
            inr::InrMapMode::LengthAngle => PhysicsMapMode::LengthAngle,
            inr::InrMapMode::Yx => PhysicsMapMode::YX,
            inr::InrMapMode::AngleLength => PhysicsMapMode::AngleLength,
        },
        gravity: p.gravity,
        length: p.length,
        frequency: p.frequency,
        angle_damping: p.angle_damping,
        length_damping: p.length_damping,
        output_scale: p.output_scale,
        local_only: p.local_only,
    });

    let asset = InxNode {
        uuid: n.uuid,
        name: n.name.clone().into_boxed_str(),
        node_type,
        material,
        transform: Transform {
            translation: n.translation.into(),
            rotation: Quat::from_euler(EulerRot::XYZ, n.rotation[0], n.rotation[1], n.rotation[2]),
            scale: Vec3::new(n.scale[0], n.scale[1], 1.0),
        },
        zsort: n.zsort,
        enabled: n.enabled,
        physics_data,
        children: child_nodes,
    };

    named_nodes.insert(asset.name.clone(), asset.clone());
    nodes.push(asset.clone());
    Ok(asset)
}

fn tex_handle(images: &[Handle<Image>], index: i32) -> Option<Handle<Image>> {
    usize::try_from(index).ok().and_then(|i| images.get(i).cloned())
}

fn convert_mesh(model: &InrModel, m: &inr::InrMesh) -> Result<InxMesh, inr::InrError> {
    let positions = model.view_f32(m.positions)?;
    let uvs = model.view_f32(m.uvs)?;
    let indices = model.view_u32(m.indices)?;
    let count = (m.vertex_count as usize).min(positions.len() / 2).min(uvs.len() / 2);

    let mut vertex_buffer = Vec::with_capacity(count);
    let mut uv_buffer = Vec::with_capacity(count);
    for i in 0..count {
        // Puppet space is Y-down; flip to Bevy's Y-up here (same as INX path).
        vertex_buffer.push([positions[i * 2], -positions[i * 2 + 1]]);
        uv_buffer.push([uvs[i * 2], uvs[i * 2 + 1]]);
    }

    Ok(InxMesh {
        vertex_buffer,
        uv_buffer,
        index_buffer: indices,
        origin: m.origin.into(),
    })
}

fn convert_masks(doc: &inr::InrDoc, masks: &[inr::InrMask]) -> Vec<InxMask> {
    masks
        .iter()
        .filter_map(|m| {
            let source = doc.nodes.get(m.node as usize)?;
            Some(InxMask {
                source_uuid: source.uuid,
                mode: match m.mode {
                    inr::InrMaskMode::Dodge => InxMaskMode::Dodge,
                    inr::InrMaskMode::Mask => InxMaskMode::Mask,
                },
            })
        })
        .collect()
}

fn convert_param(model: &InrModel, p: &inr::InrParam) -> Result<InxParam, inr::InrError> {
    let doc = &model.doc;
    let mut bindings = Vec::with_capacity(p.bindings.len());
    for b in &p.bindings {
        let Some(node) = doc.nodes.get(b.node as usize) else {
            continue;
        };
        let x = (b.x_count as usize).max(1);
        let y = (b.y_count as usize).max(1);

        let values = match b.kind {
            inr::InrBindingKind::Scalar => {
                let data = model.view_f32(b.view)?;
                let values_per_frame = (data.len() / x).max(1);
                InxBindingValues::Transform(InxFlatTransform {
                    data,
                    frames: x,
                    values_per_frame,
                })
            }
            inr::InrBindingKind::Deform => {
                let flat = model.view_f32(b.view)?;
                let data: Vec<[f32; 2]> =
                    flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
                let vertices_per_frame = (data.len() / x).max(1);
                InxBindingValues::Deform(InxFlatDeform {
                    data,
                    frames: x,
                    vertices_per_frame,
                })
            }
            inr::InrBindingKind::Other => InxBindingValues::Other,
        };

        // Flattened row-major [x][y] back to per-row vectors.
        let mut is_set: Vec<Vec<bool>> = b.is_set.chunks(y).map(|c| c.to_vec()).collect();
        is_set.resize(x, vec![false; y]);

        bindings.push(InxBinding {
            node_uuid: node.uuid,
            param_name: b.target.into(),
            interpolation: b.interpolation.into(),
            values,
            is_set,
        });
    }

    Ok(InxParam {
        uuid: p.uuid,
        name: p.name.clone(),
        is_vec2: p.is_vec2,
        min: p.min,
        max: p.max,
        defaults: p.defaults,
        axis_points: p.axis_points.clone(),
        merge_mode: p.merge_mode.into(),
        bindings,
    })
}

fn convert_animation(doc: &inr::InrDoc, a: &inr::InrAnimation) -> InxAnimation {
    let lanes = a
        .lanes
        .iter()
        .filter_map(|l| {
            let param = doc.params.get(usize::try_from(l.param).ok()?)?;
            Some(InxAnimationLane {
                param_uuid: param.uuid,
                target: l.target,
                interpolation: l.interpolation.into(),
                merge_mode: l.merge_mode.into(),
                keyframes: l
                    .keyframes
                    .iter()
                    .map(|k| InxKeyframe {
                        frame: k[0] as u32,
                        value: k[1],
                        tension: k[2],
                    })
                    .collect(),
            })
        })
        .collect();

    InxAnimation {
        name: a.name.clone(),
        duration: a.length as f32 * a.timestep,
        timestep: a.timestep,
        lanes,
    }
}

// --- INR enum -> runtime enum mappings --------------------------------------

impl From<inr::InrBlendMode> for BlendMode {
    fn from(b: inr::InrBlendMode) -> Self {
        use inr::InrBlendMode as B;
        match b {
            B::Normal => Self::Normal,
            B::Multiply => Self::Multiply,
            B::Screen => Self::Screen,
            B::Overlay => Self::Overlay,
            B::Darken => Self::Darken,
            B::Lighten => Self::Lighten,
            B::ColorDodge => Self::ColorDodge,
            B::LinearDodge => Self::LinearDodge,
            B::Add => Self::Add,
            B::ColorBurn => Self::ColorBurn,
            B::HardLight => Self::HardLight,
            B::SoftLight => Self::SoftLight,
            B::Subtract => Self::Subtract,
            B::Difference => Self::Difference,
            B::Exclusion => Self::Exclusion,
            B::Inverse => Self::Inverse,
            B::DestinationIn => Self::DestinationIn,
            B::ClipToLower => Self::ClipToLower,
            B::SliceFromLower => Self::SliceFromLower,
        }
    }
}

impl From<inr::InrMergeMode> for InxMergeMode {
    fn from(m: inr::InrMergeMode) -> Self {
        use inr::InrMergeMode as M;
        match m {
            M::Additive => Self::Additive,
            M::Multiplicative => Self::Multiply,
            M::Override => Self::Override,
            M::Forced => Self::Forced,
        }
    }
}

impl From<inr::InrInterpolation> for InxInterpolation {
    fn from(i: inr::InrInterpolation) -> Self {
        use inr::InrInterpolation as I;
        match i {
            I::Linear => Self::Linear,
            I::Stepped | I::Nearest => Self::Stepped,
            I::Cubic => Self::Cubic,
        }
    }
}

impl From<inr::InrBindingTarget> for InxParamName {
    fn from(t: inr::InrBindingTarget) -> Self {
        use inr::InrBindingTarget as T;
        match t {
            T::TranslateX => Self::TransformTX,
            T::TranslateY => Self::TransformTY,
            T::TranslateZ => Self::TransformTZ,
            T::ScaleX => Self::TransformSX,
            T::ScaleY => Self::TransformSY,
            T::RotateX => Self::TransformRX,
            T::RotateY => Self::TransformRY,
            T::RotateZ => Self::TransformRZ,
            T::Deform => Self::Deform,
            T::Opacity => Self::Opacity,
            T::Other => Self::Other,
        }
    }
}
