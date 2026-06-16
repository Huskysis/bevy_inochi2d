//! Offscreen color + depth/stencil targets for composite layers.
//!
//! A [`CompositeFramebufferEntry`] is allocated per composite nesting level
//! and reused across frames; the render node draws children into the
//! offscreen target and then samples it back through the composite bind
//! group to blend onto the parent target.

use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::RenderDevice,
    },
};


use super::*;

pub struct CompositeFramebufferEntry {
    pub albedo: Texture,
    pub albedo_view: TextureView,
    pub depth_stencil: Texture,
    pub depth_stencil_view: TextureView,
    pub bindgroup: BindGroup,
}

impl CompositeFramebufferEntry {
    pub fn new(device: &RenderDevice, size: UVec2, pipeline: &InxPipeline, index: usize) -> Self {
        let extent = Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        };

        let albedo = device.create_texture(&TextureDescriptor {
            label: Some(&format!("inx_cf_albedo_{index}")),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let albedo_view = albedo.create_view(&TextureViewDescriptor::default());

        let depth_stencil = device.create_texture(&TextureDescriptor {
            label: Some(&format!("inx_cf_depth_stencil_{index}")),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth24PlusStencil8,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_stencil_view = depth_stencil.create_view(&TextureViewDescriptor::default());

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some(&format!("inx_cf_sampler_{index}")),
            address_mode_u: AddressMode::ClampToBorder,
            address_mode_v: AddressMode::ClampToBorder,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            border_color: Some(bevy::image::ImageSamplerBorderColor::TransparentBlack.into()),
            ..Default::default()
        });

        let bindgroup = device.create_bind_group(
            format!("inx_cf_bindgroup_{index}").as_str(),
            &pipeline.texture_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&albedo_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        );

        Self {
            albedo,
            albedo_view,
            depth_stencil,
            depth_stencil_view,
            bindgroup,
        }
    }
}

const MAX_COMPOSITE_DEPTH: usize = 4;

/// Per-view offscreen resources: composite pool and scene depth-stencil.
/// Each view (window camera, render-to-texture camera) has its own size and
/// sample count, so these can't be shared globally.
#[derive(Component)]
pub struct ViewInxFramebuffers {
    pub composite: CompositeFramebufferPool,
    pub scene: SceneFramebuffer,
}

pub struct CompositeFramebufferPool {
    pub entries: Vec<CompositeFramebufferEntry>,
    pub size: UVec2,

    // Shared fullscreen triangle buffers (solo necesitas un set)
    pub vertex_buffer: Buffer,
    pub uv_buffer: Buffer,
    pub index_buffer: Buffer,
}

impl CompositeFramebufferPool {
    pub fn new(device: &RenderDevice, size: UVec2, pipeline: &InxPipeline) -> Self {
        let entries = (0..MAX_COMPOSITE_DEPTH)
            .map(|i| CompositeFramebufferEntry::new(device, size, pipeline, i))
            .collect();

        // Fullscreen triangle (compartido)
        let vertex_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("inx_cf_vertex_buffer"),
            contents: bytemuck::cast_slice(&[[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]]),
            usage: BufferUsages::VERTEX,
        });

        let uv_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("inx_cf_uv_buffer"),
            contents: bytemuck::cast_slice(&[[0.0f32, 1.0], [2.0, 1.0], [0.0, -1.0]]),
            usage: BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("inx_cf_index_buffer"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        Self {
            entries,
            size,
            vertex_buffer,
            uv_buffer,
            index_buffer,
        }
    }

    pub fn resize(&mut self, device: &RenderDevice, new_size: UVec2, pipeline: &InxPipeline) {
        if self.size != new_size {
            self.entries = (0..MAX_COMPOSITE_DEPTH)
                .map(|i| CompositeFramebufferEntry::new(device, new_size, pipeline, i))
                .collect();
            self.size = new_size;
            // vertex/uv/index no cambian con resize
        }
    }
}

/// Depth-stencil attachment matching the ViewTarget, used for stencil masks
/// when rendering directly into the view.
pub struct SceneFramebuffer {
    pub depth_stencil: Texture,
    pub depth_stencil_view: TextureView,
    pub size: UVec2,
    pub samples: u32,
}

impl SceneFramebuffer {
    pub fn new(device: &RenderDevice, size: UVec2, samples: u32) -> Self {
        let extent = Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        };

        let depth_stencil = device.create_texture(&TextureDescriptor {
            label: Some("inx_scene_depth_stencil"),
            size: extent,
            mip_level_count: 1,
            sample_count: samples,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth24PlusStencil8,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_stencil_view = depth_stencil.create_view(&TextureViewDescriptor {
            label: Some("inx_scene_depth_stencil_view"),
            ..Default::default()
        });

        Self {
            depth_stencil,
            depth_stencil_view,
            size,
            samples,
        }
    }

    pub fn resize(&mut self, device: &RenderDevice, new_size: UVec2, samples: u32) {
        if self.size != new_size || self.samples != samples {
            *self = Self::new(device, new_size, samples);
        }
    }
}

