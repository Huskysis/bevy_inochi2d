use bevy::{
    asset::load_embedded_asset,
    mesh::VertexBufferLayout,
    platform::collections::HashMap,
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        texture::{FallbackImage, GpuImage},
        view::{
            ExtractedView, ViewTarget, ViewUniform, ViewUniforms,
        },
    },
};

use crate::BlendMode;

use super::*;

#[derive(ShaderType, Component, Clone, Debug, Default)]
pub struct InxUniform {
    transform: Mat4,
    offset: Vec2,
    opacity: f32,
    mask_threshold: f32,
    emissive_strength: f32,
    tint: Vec3,
    screen_tint: Vec3,
}

impl InxUniform {
    pub fn new(
        transform: Mat4,
        offset: Vec2,
        tint: Vec3,
        screen_tint: Vec3,
        opacity: f32,
        emissive_strength: f32,
        mask_threshold: f32,
    ) -> Self {
        Self {
            transform,
            offset,
            tint,
            screen_tint,
            opacity,
            emissive_strength,
            mask_threshold,
        }
    }
}

#[derive(ShaderType, Component, Clone, Debug, Default)]
pub struct CompositeUniform {
    transform: Mat4,
    opacity: f32,
    tint: Vec3,
    screen_tint: Vec3,
}

impl CompositeUniform {
    pub fn new(transform: Mat4, opacity: f32, tint: Vec3, screen_tint: Vec3) -> Self {
        Self {
            transform,
            opacity,
            tint,
            screen_tint,
        }
    }
}

#[derive(Component)]
pub struct ViewBindGroupInx {
    pub(crate) value: BindGroup,
}

/// Buffers GPU del puppet
#[derive(Component)]
pub struct PuppetGpuBuffers {
    // Buffer de vertices (posiciones, UVs, deformaciones)
    // Muy complicado de manejar para mi
    // pub interleaved_buffer: bevy::render::render_resource::Buffer,
    /// Buffer de vertices (posiciones)
    pub vertex_buffer: bevy::render::render_resource::Buffer,
    /// Buffer de UVs
    pub uv_buffer: bevy::render::render_resource::Buffer,
    /// Buffer de deformaciones (actualizado cada frame)
    pub deform_buffer: bevy::render::render_resource::Buffer,
    /// Buffer de indices
    pub index_buffer: bevy::render::render_resource::Buffer,
}

/// Resource con las texturas del modelo cargadas
/// BindGroups de textura compartidos globalmente.
/// Key = AssetId<Image>, no Entity.
/// Si 10 puppets usan la misma textura, hay 1 BindGroup (no 10).
#[derive(Resource, Default)]
pub struct InxTexturesBindGroup {
    /// AssetId - BindGroup (compartido entre todos los puppets)
    pub by_asset: HashMap<AssetId<Image>, BindGroup>,

    /// Fallback para texturas no cargadas o slots vacios (u32::MAX)
    pub fallback: Option<BindGroup>,

    /// Per-entity: mapea local texture index - AssetId
    /// Para que el render pueda resolver part.textures[n] - BindGroup
    pub entity_maps: HashMap<Entity, Vec<AssetId<Image>>>,
}

#[derive(Resource)]
pub struct InxPipeline {
    /// Keyed by (blend mode, sample count). Sample count 1 is used for
    /// offscreen composite content; the view's sample count for direct
    /// rendering into the ViewTarget.
    pub basic_pipeline: HashMap<(BlendMode, u32), CachedRenderPipelineId>,
    pub composite_pipeline: HashMap<(BlendMode, u32), CachedRenderPipelineId>,
    pub mask_pipeline: HashMap<u32, CachedRenderPipelineId>,

    pub view_layout: BindGroupLayout,
    pub basic_uniform_layout: BindGroupLayout,
    pub composite_uniform_layout: BindGroupLayout,
    pub texture_layout: BindGroupLayout,

    // Kept for lazy per-sample-count pipeline creation.
    shader_basic: Handle<Shader>,
    shader_mask: Handle<Shader>,
    shader_composite: Handle<Shader>,
    basic_layouts: Vec<BindGroupLayoutDescriptor>,
    composite_layouts: Vec<BindGroupLayoutDescriptor>,
    mask_layouts: Vec<BindGroupLayoutDescriptor>,
}

impl InxPipeline {
    /// Create the pipeline set for `samples` if it doesn't exist yet.
    pub fn ensure_samples(&mut self, samples: u32, pipeline_cache: &PipelineCache) {
        if self.mask_pipeline.contains_key(&samples) {
            return;
        }
        for (mode, id) in create_part_pipeline(
            &self.shader_basic,
            false,
            &self.basic_layouts,
            pipeline_cache,
            samples,
        ) {
            self.basic_pipeline.insert((mode, samples), id);
        }
        for (mode, id) in create_part_pipeline(
            &self.shader_composite,
            true,
            &self.composite_layouts,
            pipeline_cache,
            samples,
        ) {
            self.composite_pipeline.insert((mode, samples), id);
        }
        self.mask_pipeline.insert(
            samples,
            create_stencil_pipeline(&self.shader_mask, &self.mask_layouts, pipeline_cache, samples),
        );
    }
}

impl FromWorld for InxPipeline {
    fn from_world(world: &mut bevy::ecs::world::World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let assets = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let view_layout_desc = BindGroupLayoutDescriptor::new(
            "Inx Pipeline View Layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            }],
        );
        let view_layout =
            render_device.create_bind_group_layout("Inx Pipeline View Layout", &view_layout_desc.entries);

        let basic_uniform_layout_desc = BindGroupLayoutDescriptor::new(
            "Inx Pipeline Uniform Layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(InxUniform::min_size()),
                },
                count: None,
            }],
        );
        let basic_uniform_layout = render_device
            .create_bind_group_layout("Inx Pipeline Uniform Layout", &basic_uniform_layout_desc.entries);

        let composite_uniform_layout_desc = BindGroupLayoutDescriptor::new(
            "Inx Pipeline Composite Uniform Layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(CompositeUniform::min_size()),
                },
                count: None,
            }],
        );
        let composite_uniform_layout = render_device.create_bind_group_layout(
            "Inx Pipeline Composite Uniform Layout",
            &composite_uniform_layout_desc.entries,
        );

        let texture_layout_desc = BindGroupLayoutDescriptor::new(
            "Inx Pipeline Texture Layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        );
        let texture_layout = render_device
            .create_bind_group_layout("Inx Pipeline Texture Layout", &texture_layout_desc.entries);

        let shader_basic = load_embedded_asset!(assets, "../basic.wgsl");
        let shader_mask = load_embedded_asset!(assets, "../mask.wgsl");
        let shader_composite = load_embedded_asset!(assets, "../composite.wgsl");

        let basic_bind_group_layouts = vec![
            view_layout_desc.clone(),
            basic_uniform_layout_desc.clone(),
            texture_layout_desc.clone(),
            texture_layout_desc.clone(),
            texture_layout_desc.clone(),
        ];
        let composite_bind_group_layouts = vec![
            view_layout_desc.clone(),
            composite_uniform_layout_desc.clone(),
            texture_layout_desc.clone(),
            texture_layout_desc.clone(),
            texture_layout_desc.clone(),
        ];

        let bind_group_layout_mask = vec![
            view_layout_desc.clone(),
            basic_uniform_layout_desc.clone(),
            texture_layout_desc.clone(),
        ];

        let mut this = Self {
            basic_pipeline: HashMap::default(),
            composite_pipeline: HashMap::default(),
            mask_pipeline: HashMap::default(),

            view_layout,
            basic_uniform_layout,
            composite_uniform_layout,
            texture_layout,

            shader_basic,
            shader_mask,
            shader_composite,
            basic_layouts: basic_bind_group_layouts,
            composite_layouts: composite_bind_group_layouts,
            mask_layouts: bind_group_layout_mask,
        };
        // Offscreen (composite) content always renders at sample count 1.
        this.ensure_samples(1, pipeline_cache);
        this
    }
}

fn create_part_pipeline(
    shader: &Handle<Shader>,
    composite: bool,
    layout: &[BindGroupLayoutDescriptor],
    pipeline_cache: &PipelineCache,
    samples: u32,
) -> HashMap<BlendMode, CachedRenderPipelineId> {
    let vertex_buffers = {
        let capacity = if composite { 2 } else { 3 };
        let mut vb = Vec::with_capacity(capacity as usize);
        (0..capacity).for_each(|idx| {
            vb.push(VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: idx,
                }],
            })
        });
        vb
    };
    let mut basic = HashMap::default();
    for blend_mode in BlendMode::ALL {
        let label = format!(
            "inx_pipeline_{}_{:?}_x{samples}",
            if composite { "composite" } else { "part" },
            blend_mode
        );

        let targets = vec![Some(ColorTargetState {
            format: TextureFormat::Rgba8UnormSrgb,
            blend: Some(blend_mode.blend_state()),
            write_mask: ColorWrites::ALL,
        })];

        let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: layout.to_owned(),
            vertex: VertexState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: None,
                buffers: vertex_buffers.clone(),
            },
            fragment: Some(FragmentState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: None,
                targets,
            }),
            depth_stencil: if composite {
                None
            } else {
                Some(DepthStencilState {
                    format: TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Always,
                    stencil: StencilState {
                        front: StencilFaceState {
                            compare: CompareFunction::Equal,
                            fail_op: StencilOperation::Keep,
                            depth_fail_op: StencilOperation::Keep,
                            pass_op: StencilOperation::Keep,
                        },
                        back: StencilFaceState {
                            compare: CompareFunction::Equal,
                            fail_op: StencilOperation::Keep,
                            depth_fail_op: StencilOperation::Keep,
                            pass_op: StencilOperation::Keep,
                        },
                        read_mask: 0xff,
                        write_mask: 0x0,
                    },
                    bias: DepthBiasState::default(),
                })
            },
            multisample: MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            primitive: PrimitiveState {
                ..Default::default()
            },
            ..Default::default()
        });

        basic.insert(blend_mode, pipeline);
    }

    basic
}

fn create_stencil_pipeline(
    shader: &Handle<Shader>,
    layout: &[BindGroupLayoutDescriptor],
    pipeline_cache: &PipelineCache,
    samples: u32,
) -> CachedRenderPipelineId {
    pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some(format!("inx_pipeline_stencil_x{samples}").into()),
        layout: layout.to_owned(),
        vertex: VertexState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: None,
            buffers: vec![
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vec2>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                },
                VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vec2>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 1,
                    }],
                },
            ],
        },
        fragment: Some(FragmentState {
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: None,
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8UnormSrgb,
                blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: ColorWrites::empty(),
            })],
        }),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth24PlusStencil8,
            depth_write_enabled: false,
            depth_compare: CompareFunction::Always,
            stencil: StencilState {
                front: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail_op: StencilOperation::Keep,
                    depth_fail_op: StencilOperation::Keep,
                    pass_op: StencilOperation::Replace,
                },
                back: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail_op: StencilOperation::Keep,
                    depth_fail_op: StencilOperation::Keep,
                    pass_op: StencilOperation::Replace,
                },
                read_mask: 0xff,
                write_mask: 0xff,
            },
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        ..Default::default()
    })
}

pub fn prepare_puppet_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    puppets: Query<(Entity, &InxData), Without<PuppetGpuBuffers>>,
) {
    for (entity, extracted) in puppets.iter() {
        // Crear vertex buffer
        let vertex_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("inx_puppet_vertex_buffer"),
            contents: bytemuck::cast_slice(&extracted.verts),
            usage: BufferUsages::VERTEX,
        });

        // Crear UV buffer
        let uv_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("inx_puppet_uv_buffer"),
            contents: bytemuck::cast_slice(&extracted.uvs),
            usage: BufferUsages::VERTEX,
        });

        // Crear deform buffer (DYNAMIC porque se actualiza cada frame)
        let deform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("inx_puppet_deform_buffer"),
            contents: bytemuck::cast_slice(&extracted.deforms),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        // Crear index buffer
        let index_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("inx_puppet_index_buffer"),
            contents: bytemuck::cast_slice(&extracted.indices),
            usage: BufferUsages::INDEX,
        });

        commands.entity(entity).insert(PuppetGpuBuffers {
            vertex_buffer,
            uv_buffer,
            deform_buffer,
            index_buffer,
        });
    }
}

pub fn prepare_view_target_composite_scene(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    mut views: Query<(Entity, &ViewTarget, Option<&mut ViewInxFramebuffers>)>,
    mut pipeline: ResMut<InxPipeline>,
) {
    for (entity, view, framebuffers) in views.iter_mut() {
        let size = view.main_texture().size();
        let viewport_size = UVec2::new(size.width, size.height);
        // main_texture() is the resolve target (1x); MSAA texture is separate
        let samples = view
            .sampled_main_texture()
            .map(|t| t.sample_count())
            .unwrap_or(1);

        pipeline.ensure_samples(samples, &pipeline_cache);

        if let Some(mut fbs) = framebuffers {
            fbs.composite.resize(&render_device, viewport_size, &pipeline);
            fbs.scene.resize(&render_device, viewport_size, samples);
        } else {
            commands.entity(entity).insert(ViewInxFramebuffers {
                composite: CompositeFramebufferPool::new(&render_device, viewport_size, &pipeline),
                scene: SceneFramebuffer::new(&render_device, viewport_size, samples),
            });
        }
    }
}

pub fn prepare_texture_bind_group(
    render_device: Res<RenderDevice>,
    mut textures: ResMut<InxTexturesBindGroup>,
    inx_pipeline: Res<InxPipeline>,
    fallback_img: Res<FallbackImage>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    query: Query<(Entity, &InxData)>,
) {
    let fallback = &fallback_img.d2;

    // Crear fallback bind group una vez
    if textures.fallback.is_none() {
        textures.fallback = Some(render_device.create_bind_group(
            Some("inx_texture_fallback"),
            &inx_pipeline.texture_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&fallback.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&fallback.sampler),
                },
            ],
        ));
    }

    for (entity, extract) in query.iter() {
        // Omite si este entidad ya tiene su mapa registrado
        if textures.entity_maps.contains_key(&entity) {
            continue;
        }

        if extract.textures.is_empty() {
            textures.entity_maps.insert(entity, Vec::new());
            continue;
        }

        // Registrar el mapa local para este entidad
        let asset_ids: Vec<AssetId<Image>> = extract.textures.clone();
        textures.entity_maps.insert(entity, asset_ids);

        // Crear bind groups solo para texturas NUEVAS
        for &asset_id in &extract.textures {
            if textures.by_asset.contains_key(&asset_id) {
                continue; // Ya existe - compartido con otro puppet
            }

            let gpu_texture = gpu_images.get(asset_id).unwrap_or(fallback);

            let bind_group = render_device.create_bind_group(
                Some("inx_texture_shared"),
                &inx_pipeline.texture_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&gpu_texture.texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&gpu_texture.sampler),
                    },
                ],
            );

            textures.by_asset.insert(asset_id, bind_group);
        }
    }
}

pub fn prepare_inx_view_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    inx_pipeline: Res<InxPipeline>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<Entity, With<ExtractedView>>,
) {
    let Some(binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    for view_entity in views.iter() {
        commands.entity(view_entity).insert(ViewBindGroupInx {
            value: render_device.create_bind_group(
                Some("inx_view_binding_group"),
                &inx_pipeline.view_layout,
                &[BindGroupEntry {
                    binding: 0,
                    resource: binding.clone(),
                }],
            ),
        });
    }
}

pub fn update_deform_buffer(
    render_queue: Res<RenderQueue>,
    mut query: Query<(&mut InxData, &PuppetGpuBuffers), Changed<InxData>>,
) {
    for (mut data, gpu) in query.iter_mut() {
        let Some((start, end)) = data.deform_dirty.take() else {
            // Sin cambios omite write_buffer
            continue;
        };

        if data.deforms.is_empty() {
            continue;
        }

        let start = start as usize;
        let end = end.min((data.deforms.len() * std::mem::size_of::<[f32; 2]>()) as u32) as usize;

        if start >= end {
            continue;
        }

        // Solo escribir el rango que cambio
        let all_bytes: &[u8] = bytemuck::cast_slice(&data.deforms);
        render_queue.write_buffer(&gpu.deform_buffer, start as u64, &all_bytes[start..end]);
    }
}

