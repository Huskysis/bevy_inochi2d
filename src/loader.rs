use std::io;

use bevy::{
    asset::{AssetLoader, LoadContext, RenderAssetUsages},
    platform::collections::HashMap,
    reflect::TypePath,
    utils::default,
};
use bevy_image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use inochi2d_parser::owned::{NodeDataType, Puppet as RawPuppet};
use thiserror::Error;

use crate::*;

#[derive(Debug, Error)]
pub enum InxError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Parse error: {0}")]
    Parse(String),
}

#[derive(TypePath)]
pub struct InxLoader;

impl AssetLoader for InxLoader {
    type Asset = InxPuppet;

    type Settings = ();

    type Error = InxError;

    async fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;

        let puppet = inochi2d_parser::owned::Puppet::from_bytes(&bytes)?;
        convert_puppet(puppet, load_context)
    }

    fn extensions(&self) -> &[&str] {
        &["inx", "inp"]
    }
}

fn convert_puppet(raw: RawPuppet, ctx: &mut LoadContext<'_>) -> Result<InxPuppet, InxError> {
    let mut textures = Vec::new();
    let mut nodes = Vec::new();
    let mut params = Vec::new();
    let mut animations = Vec::new();

    let mut named_nodes = HashMap::default();
    let mut named_params = HashMap::default();
    let mut named_animations = HashMap::default();

    for tex in &raw.textures {
        let image = convert_texture(tex)?;
        let label = format!("texture_{}", tex.id);
        let handle = ctx.add_labeled_asset(label, image);

        textures.push(handle);
    }

    // Nodos recursivo
    convert_node_tree(&raw.nodes, &textures, &mut nodes, &mut named_nodes)?;

    // Params
    for (uuid, param) in &raw.params {
        let asset = convert_param(param);
        let label = format!("param_{}", uuid);
        let handle = ctx.add_labeled_asset(label.clone(), asset);

        named_params.insert(param.name.clone().into_boxed_str(), handle.clone());
        params.push(handle);
    }

    // Animations
    for (name, anim) in &raw.animations {
        let asset = convert_animation(anim);
        let label = format!("animation_{}", name);
        let handle = ctx.add_labeled_asset(label, asset);

        named_animations.insert(name.clone().into_boxed_str(), handle.clone());
        animations.push(handle);
    }

    // Meta
    let meta = InxMeta {
        name: raw.meta.name.unwrap_or_default(),
        version: raw.meta.version,
        rigger: raw.meta.rigger.unwrap_or_default(),
        artist: raw.meta.artist.unwrap_or_default(),
        license_url: raw.meta.license_url.unwrap_or_default(),
        copyright: raw.meta.copyright.unwrap_or_default(),
        contact: raw.meta.contact.unwrap_or_default(),
        rights: raw.meta.rights.unwrap_or_default(),
        reference: raw.meta.reference.unwrap_or_default(),
        thumbnail_id: raw.meta.thumbnail_id,
        preserve_pixels: raw.meta.preserve_pixels,
    };

    let physics = InxPhysics {
        pixels_per_meter: raw.physics.pixels_per_meter,
        gravity: raw.physics.gravity,
    };

    Ok(InxPuppet {
        nodes,
        textures,
        params,
        animations,
        named_nodes,
        named_params,
        named_animations,
        meta,
        physics,
        // source: Some(raw),
        source: None,
    })
}

fn convert_texture(
    texture: &inochi2d_parser::owned::Texture,
) -> Result<bevy::image::Image, InxError> {
    use bevy::image::{Image, ImageFormat, ImageType};

    let data = match &texture.data {
        inochi2d_parser::owned::TextureData::Encoded(items)
        | inochi2d_parser::owned::TextureData::Rgba(items) => items.clone(),
    };

    let format = match texture.format {
        inochi2d_parser::owned::TextureFormat::Png => ImageFormat::Png,
        inochi2d_parser::owned::TextureFormat::Tga => ImageFormat::Tga,
        inochi2d_parser::owned::TextureFormat::Bc7 => {
            return Err(InxError::Parse("BC7 not supported".to_string()));
        }
    };

    let image = Image::from_buffer(
        &data,
        ImageType::Format(format),
        default(),
        false, // debe ser false
        // default(),
        ImageSampler::Descriptor(ImageSamplerDescriptor {
            label: Some(format!("texture_{}", texture.id)),
            address_mode_u: ImageAddressMode::ClampToBorder,
            address_mode_v: ImageAddressMode::ClampToBorder,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            mipmap_filter: ImageFilterMode::Linear,
            ..Default::default()
        }),
        RenderAssetUsages::RENDER_WORLD,
    )
    .map_err(|e| InxError::Parse(e.to_string()))?;

    Ok(image)
}

fn convert_node_tree(
    node: &inochi2d_parser::prelude::Node,
    images: &[Handle<Image>],
    nodes: &mut Vec<InxNode>,
    named_nodes: &mut HashMap<Box<str>, InxNode>,
) -> Result<InxNode, InxError> {
    // Recursivo para hijos
    let mut children_handles = Vec::new();

    for child in &node.children {
        let handle = convert_node_tree(child, images, nodes, named_nodes)?;
        children_handles.push(handle);
    }

    let material_handle = convert_material(&node.type_node, images)?;

    let node_type = match &node.type_node {
        NodeDataType::Part(_) => InxNodeType::Part,
        NodeDataType::Composite(_) => InxNodeType::Composite,
        NodeDataType::Mask(_) => InxNodeType::Mask,
        NodeDataType::MeshGroup(_) => InxNodeType::MeshGroup,
        NodeDataType::Camera(_) => InxNodeType::Camera,
        NodeDataType::SimplePhysics(_) => InxNodeType::SimplePhysics,
        NodeDataType::Generic => InxNodeType::Generic,
    };

    let physics_data = match &node.type_node {
        NodeDataType::SimplePhysics(data) => Some(SimplePhysicsConfig {
            param_uuid: data.param,
            model: data.model_type.into(),
            map_mode: data.map_mode.into(),
            gravity: data.gravity,
            length: data.length,
            frequency: data.frequency,
            angle_damping: data.angle_damping,
            length_damping: data.length_damping,
            output_scale: data.output_scale,
            local_only: data.local_only.unwrap_or(false),
        }),
        _ => None,
    };

    let transform = convert_transform(&node.transform);

    let asset = InxNode {
        uuid: node.uuid,
        name: node.name.clone().into_boxed_str(),
        node_type,
        material: material_handle,
        transform,
        zsort: node.zsort,
        enabled: node.enabled,
        physics_data,
        children: children_handles,
    };

    named_nodes.insert(node.name.clone().into_boxed_str(), asset.clone());
    nodes.push(asset.clone());

    Ok(asset)
}

fn convert_material(
    data: &NodeDataType,
    images: &[Handle<Image>],
) -> Result<Option<InxMaterial>, InxError> {
    let material = match data {
        NodeDataType::Part(part_data) => {
            let tex_albedo = part_data.textures[0];
            let tex_emissive = part_data.textures[1];
            let tex_bumpmap = part_data.textures[2];
            Some(InxMaterial {
                mesh: convert_inx_mesh(&part_data.mesh),
                texture_albedo: images.get(tex_albedo as usize).cloned(),
                texture_emissive: images.get(tex_emissive as usize).cloned(),
                texture_bumpmap: images.get(tex_bumpmap as usize).cloned(),
                textures: part_data.textures,
                tint: part_data.tint.into(),
                screen_tint: part_data.screen_tint.into(),
                opacity: part_data.opacity,
                emissive_strength: part_data.emission_strength,
                mask_threshold: part_data.mask_threshold,
                blend_mode: part_data.blend_mode.into(),
                masks: part_data
                    .mask
                    .iter()
                    .map(|m| InxMask {
                        source_uuid: m.source,
                        mode: m.mode.into(),
                    })
                    .collect(),
            })
        }
        NodeDataType::Camera(_camera_data) => None,
        NodeDataType::SimplePhysics(_simple_physics_data) => None,
        NodeDataType::Composite(composite_data) => Some(InxMaterial {
            mesh: None,
            texture_albedo: None,
            texture_emissive: None,
            texture_bumpmap: None,
            textures: [u32::MAX; 3],
            tint: composite_data.tint.into(),
            screen_tint: composite_data.screen_tint.into(),
            opacity: composite_data.opacity,
            emissive_strength: 0.0,
            mask_threshold: composite_data.mask_threshold,
            blend_mode: composite_data.blend_mode.into(),
            masks: composite_data
                .mask
                .iter()
                .map(|m| InxMask {
                    source_uuid: m.source,
                    mode: m.mode.into(),
                })
                .collect(),
        }),
        NodeDataType::Mask(_mask_data) => None,
        NodeDataType::MeshGroup(_mesh_group_data) => None,
        NodeDataType::Generic => None,
    };

    Ok(material)
}

fn convert_inx_mesh(raw: &Option<inochi2d_parser::prelude::Mesh>) -> Option<Arc<InxMesh>> {
    let Some(raw) = raw else {
        return None;
    };
    let vertex_count = raw.vertices.len() / 2;

    let mut position = Vec::with_capacity(vertex_count);
    let mut uv = Vec::with_capacity(vertex_count);

    for i in 0..vertex_count {
        let px = raw.vertices[i * 2];
        let py = -raw.vertices[i * 2 + 1];
        let u = raw.uvs[i * 2];
        let v = raw.uvs[i * 2 + 1];
        // position.push([px, py, 0.0]);
        position.push([px, py]);
        uv.push([u, v]);
    }

    Some(Arc::new(InxMesh {
        vertex_buffer: position,
        uv_buffer: uv,
        index_buffer: raw.indices.clone(),
        origin: raw.origin.into(),
    }))
}

fn convert_transform(raw: &inochi2d_parser::prelude::Transform) -> Transform {
    Transform {
        translation: raw.translation.into(),
        rotation: {
            let r = raw.rotation;
            Quat::from_euler(EulerRot::XYZ, r[0], r[1], r[2])
        },
        scale: {
            let s = raw.scale;
            Vec3::new(s[0], s[1], 1.0)
        },
    }
}

fn convert_param(raw: &inochi2d_parser::prelude::Param) -> InxParam {
    let bindings = raw
        .bindings
        .iter()
        .map(|b| {
            let values = match &b.values {
                inochi2d_parser::prelude::BindingValues::Transform(flat) => {
                    InxBindingValues::Transform(InxFlatTransform {
                        data: flat.data.clone(),
                        frames: flat.frames,
                        values_per_frame: flat.values_per_frame,
                    })
                }
                inochi2d_parser::prelude::BindingValues::Deform(flat) => {
                    InxBindingValues::Deform(InxFlatDeform {
                        data: flat.data.clone(),
                        frames: flat.frames,
                        vertices_per_frame: flat.vertices_per_frame,
                    })
                }
                _ => InxBindingValues::Other,
            };

            InxBinding {
                node_uuid: b.node,
                param_name: convert_param_name(&b.param_name),
                interpolation: b.interpolate_mode.into(),
                values,
                is_set: b.is_set.clone(),
            }
        })
        .collect();

    InxParam {
        uuid: raw.uuid,
        name: raw.name.clone(),
        is_vec2: raw.is_vec2,
        min: raw.min,
        max: raw.max,
        defaults: raw.defaults,
        axis_points: raw.axis_points.clone(),
        merge_mode: raw.merge_mode.into(),
        bindings,
    }
}

fn convert_param_name(raw: &inochi2d_parser::prelude::ParamName) -> InxParamName {
    match raw {
        inochi2d_parser::prelude::ParamName::TransformTX => InxParamName::TransformTX,
        inochi2d_parser::prelude::ParamName::TransformTY => InxParamName::TransformTY,
        inochi2d_parser::prelude::ParamName::TransformTZ => InxParamName::TransformTZ,
        inochi2d_parser::prelude::ParamName::TransformSX => InxParamName::TransformSX,
        inochi2d_parser::prelude::ParamName::TransformSY => InxParamName::TransformSY,
        inochi2d_parser::prelude::ParamName::TransformRX => InxParamName::TransformRX,
        inochi2d_parser::prelude::ParamName::TransformRY => InxParamName::TransformRY,
        inochi2d_parser::prelude::ParamName::TransformRZ => InxParamName::TransformRZ,
        inochi2d_parser::prelude::ParamName::Deform => InxParamName::Deform,
        inochi2d_parser::prelude::ParamName::Opacity => InxParamName::Opacity,
        inochi2d_parser::prelude::ParamName::Other(_s) => InxParamName::Other,
    }
}

fn convert_animation(raw: &inochi2d_parser::prelude::Animation) -> InxAnimation {
    let lanes = raw
        .lanes
        .iter()
        .map(|l| InxAnimationLane {
            param_uuid: l.param_uuid,
            target: l.target,
            interpolation: l.interpolation.into(),
            merge_mode: l.merge_mode.into(),
            keyframes: l
                .keyframes
                .iter()
                .map(|k| InxKeyframe {
                    frame: k.frame,
                    value: k.value,
                    tension: k.tension,
                })
                .collect(),
        })
        .collect();

    InxAnimation {
        name: raw.name.clone(),
        duration: raw.duration(),
        timestep: raw.timestep,
        lanes,
    }
}
