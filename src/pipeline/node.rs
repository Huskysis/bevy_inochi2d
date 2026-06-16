//! Render-graph view node that draws a puppet for one view.
//!
//! Iterates the extracted command list, sets pipelines/bind groups for
//! parts, masks and composites, and respects [`RenderLayers`] so a given
//! view only draws puppets whose layers intersect it.

use bevy::{
    camera::visibility::RenderLayers,
    prelude::*,
    render::{
        render_graph::{RenderLabel, ViewNode},
        render_phase::TrackedRenderPass,
        render_resource::{
            encase::{UniformBuffer, private::WriteInto},
            *,
        },
        renderer::{RenderContext, RenderDevice},
        view::{
            ViewTarget, ViewUniformOffset,
        },
    },
};

use crate::BlendMode;

use super::*;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct InxViewNodeLabel;

pub struct InxRenderViewNode {
    extract_buffer: QueryState<(Entity, &'static InxData, &'static PuppetGpuBuffers)>,
    view_bindgroup: QueryState<&'static ViewBindGroupInx>,
    view_offset: QueryState<&'static ViewUniformOffset>,
}

impl FromWorld for InxRenderViewNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            extract_buffer: QueryState::new(world),
            view_bindgroup: QueryState::new(world),
            view_offset: QueryState::new(world),
        }
    }
}

impl ViewNode for InxRenderViewNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewInxFramebuffers,
        Option<&'static RenderLayers>,
    );

    fn update(&mut self, world: &mut World) {
        self.extract_buffer.update_archetypes(world);
        self.view_bindgroup.update_archetypes(world);
        self.view_offset.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext<'w>,
        (view_target, view_framebuffers, view_layers): bevy::ecs::query::QueryItem<
            'w,
            '_,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> std::result::Result<(), bevy::render::render_graph::NodeRunError> {
        let Some(inx_pipeline) = world.get_resource::<InxPipeline>() else {
            return Ok(());
        };

        let Some(pipeline_cache) = world.get_resource::<PipelineCache>() else {
            return Ok(());
        };

        if pipeline_cache.waiting_pipelines().next().is_some() {
            return Ok(());
        }

        let Some(textures) = world.get_resource::<InxTexturesBindGroup>() else {
            return Ok(());
        };

        let entity = graph.view_entity();

        let Ok(view_bindgroup) = self.view_bindgroup.get_manual(world, entity) else {
            return Ok(());
        };

        let Ok(view_offset) = self.view_offset.get_manual(world, entity) else {
            return Ok(());
        };

        let composite_fb_pool = &view_framebuffers.composite;
        let scene_fb = &view_framebuffers.scene;

        let render_device = world.resource::<RenderDevice>();

        let samples = view_target
            .sampled_main_texture()
            .map(|t| t.sample_count())
            .unwrap_or(1);

        let default_layers = RenderLayers::default();
        let view_layers = view_layers.unwrap_or(&default_layers);

        // Obtener puppets
        for (entity, data, puppet_gpu_buffers) in self.extract_buffer.iter_manual(world) {
            if data.commands.is_empty() {
                continue;
            }
            if !view_layers.intersects(&data.layers) {
                continue;
            }

            let tmp_render_pass = InxRenderPass {
                render_device,
                gpu_buffer: puppet_gpu_buffers,
                pipeline_resource: inx_pipeline,
                pipeline_cache,
                textures,
                puppet_entity: entity,
                view_bindgroup,
                view_offset,
                composite_pool: composite_fb_pool,
                scene_buffer: scene_fb,
                view_target,
                samples,
            };

            tmp_render_pass.render(render_context, data);
        }
        Ok(())
    }
}

struct CompositeFrame<'a> {
    header: &'a CompositeHeader,
    entry: &'a CompositeFramebufferEntry,
    draw_count: usize, // 0 Clear solo la primera pasada, >0 Load
}

struct UniformPool {
    /// UN solo bind group para todos los draws
    part_bind_group: BindGroup,
    /// Offset en bytes para cada DrawPart (indexado por orden de aparicion)
    part_offsets: Vec<u32>,

    /// UN solo bind group para todos los composite blits
    composite_bind_group: Option<BindGroup>,
    /// Offset en bytes para cada composite blit
    composite_offsets: Vec<u32>,

    /// UN solo bind group para todos los masks
    mask_bind_group: Option<BindGroup>,
    /// Offset en bytes para cada mask
    mask_offsets: Vec<u32>,
}

struct InxRenderPass<'r> {
    render_device: &'r RenderDevice,
    gpu_buffer: &'r PuppetGpuBuffers,

    pipeline_resource: &'r InxPipeline,
    pipeline_cache: &'r PipelineCache,

    textures: &'r InxTexturesBindGroup,
    puppet_entity: Entity,

    view_bindgroup: &'r ViewBindGroupInx,
    view_offset: &'r ViewUniformOffset,

    composite_pool: &'r CompositeFramebufferPool,
    scene_buffer: &'r SceneFramebuffer,
    view_target: &'r ViewTarget,
    samples: u32,
}

impl<'r> InxRenderPass<'r> {
    fn render(&self, render_context: &mut RenderContext, data: &InxData) {
        let mut stack: Vec<CompositeFrame> = Vec::new();
        let mut stencil_ref: u32 = 0;
        let mut composite_first_draw: Vec<bool> = Vec::new();

        let pool = self.build_uniform_pool(data);

        let mut batch: Vec<(usize, &InxPartData)> = Vec::new();
        let mut part_idx: usize = 0;
        let mut mask_idx: usize = 0;
        let mut comp_idx: usize = 0;

        for cmd in data.commands.iter() {
            match cmd {
                RenderOrder::DrawPart(part) => {
                    batch.push((part_idx, part));
                    part_idx += 1;
                }

                RenderOrder::PushMask(mask) => {
                    self.flush_batch(
                        render_context,
                        &batch,
                        &pool,
                        &stack,
                        stencil_ref,
                        &mut composite_first_draw,
                    );
                    batch.clear();

                    let (color_attachment, stencil_view, ctx_samples) =
                        if let Some(frame) = stack.last() {
                            (
                                RenderPassColorAttachment {
                                    view: &frame.entry.albedo_view,
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: Operations {
                                        load: LoadOp::Load,
                                        store: StoreOp::Store,
                                    },
                                },
                                &frame.entry.depth_stencil_view,
                                1,
                            )
                        } else {
                            (
                                self.view_target.get_color_attachment(),
                                &self.scene_buffer.depth_stencil_view,
                                self.samples,
                            )
                        };

                    stencil_ref += 1;

                    self.render_mask_pooled(
                        render_context,
                        mask,
                        stencil_view,
                        color_attachment,
                        ctx_samples,
                        stencil_ref,
                        &pool,
                        mask_idx,
                    );
                    mask_idx += 1;
                }

                RenderOrder::PopMask => {
                    self.flush_batch(
                        render_context,
                        &batch,
                        &pool,
                        &stack,
                        stencil_ref,
                        &mut composite_first_draw,
                    );
                    batch.clear();

                    stencil_ref = stencil_ref.saturating_sub(1);

                    let (color_attachment, stencil_view) = if let Some(frame) = stack.last() {
                        (
                            RenderPassColorAttachment {
                                view: &frame.entry.albedo_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: Operations {
                                    load: LoadOp::Load,
                                    store: StoreOp::Store,
                                },
                            },
                            &frame.entry.depth_stencil_view,
                        )
                    } else {
                        (
                            self.view_target.get_color_attachment(),
                            &self.scene_buffer.depth_stencil_view,
                        )
                    };

                    let _pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("inx_stencil_clear"),
                        color_attachments: &[Some(color_attachment)],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: stencil_view,
                            depth_ops: None,
                            stencil_ops: Some(Operations {
                                load: LoadOp::Clear(0),
                                store: StoreOp::Store,
                            }),
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                }

                RenderOrder::BeginComposite(header) => {
                    self.flush_batch(
                        render_context,
                        &batch,
                        &pool,
                        &stack,
                        stencil_ref,
                        &mut composite_first_draw,
                    );
                    batch.clear();

                    let depth = stack.len();
                    if depth >= self.composite_pool.entries.len() {
                        continue;
                    }

                    stack.push(CompositeFrame {
                        header,
                        entry: &self.composite_pool.entries[depth],
                        draw_count: 0,
                    });
                    composite_first_draw.push(true);
                }

                RenderOrder::EndComposite => {
                    self.flush_batch(
                        render_context,
                        &batch,
                        &pool,
                        &stack,
                        stencil_ref,
                        &mut composite_first_draw,
                    );
                    batch.clear();

                    let Some(frame) = stack.pop() else {
                        continue;
                    };
                    composite_first_draw.pop();

                    let (parent_attachment, ctx_samples) = if let Some(parent) = stack.last_mut() {
                        let first = parent.draw_count == 0;
                        parent.draw_count += 1;
                        if let Some(flag) = composite_first_draw.last_mut() {
                            *flag = false;
                        }
                        let load = if first {
                            LoadOp::Clear(LinearRgba::NONE.into())
                        } else {
                            LoadOp::Load
                        };
                        (
                            RenderPassColorAttachment {
                                view: &parent.entry.albedo_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: Operations {
                                    load,
                                    store: StoreOp::Store,
                                },
                            },
                            1,
                        )
                    } else {
                        (self.view_target.get_color_attachment(), self.samples)
                    };

                    self.render_composite_blit_pooled(
                        render_context,
                        frame.header,
                        frame.entry,
                        parent_attachment,
                        ctx_samples,
                        &pool,
                        comp_idx,
                    );
                    comp_idx += 1;
                }
            }
        }

        self.flush_batch(
            render_context,
            &batch,
            &pool,
            &stack,
            stencil_ref,
            &mut composite_first_draw,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn render_mask_pooled(
        &self,
        render_context: &mut RenderContext,
        mask: &MaskHeader,
        stencil_view: &TextureView,
        color_attachment: RenderPassColorAttachment,
        ctx_samples: u32,
        stencil_ref: u32,
        pool: &UniformPool,
        mask_idx: usize,
    ) {
        let Some(&pid) = self.pipeline_resource.mask_pipeline.get(&ctx_samples) else {
            return;
        };
        let Some(cache_pipeline) = self.pipeline_cache.get_render_pipeline(pid) else {
            return;
        };

        let color_attachments = &[Some(color_attachment)];

        let stencil_ref_value = match mask.mode {
            MaskMode::Mask => stencil_ref,
            MaskMode::Dodge => 0,
        };

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("inx_mask_pass"),
            color_attachments,
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: stencil_view,
                depth_ops: None,
                stencil_ops: Some(Operations {
                    load: LoadOp::Clear(0),
                    store: StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_stencil_reference(stencil_ref_value);
        render_pass.set_render_pipeline(cache_pipeline);
        render_pass.set_vertex_buffer(0, self.gpu_buffer.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.gpu_buffer.uv_buffer.slice(..));
        render_pass.set_index_buffer(
            self.gpu_buffer.index_buffer.slice(..),
            IndexFormat::Uint32,
        );
        render_pass.set_bind_group(0, &self.view_bindgroup.value, &[self.view_offset.offset]);

        // offset dinamico
        if let Some(ref mask_bg) = pool.mask_bind_group {
            render_pass.set_bind_group(1, mask_bg, &[pool.mask_offsets[mask_idx]]);
        }

        self.bind_texture(self.puppet_entity, &mut render_pass, 2, mask.tex_albedo);
        render_pass.draw_indexed(
            mask.index_offset..(mask.index_offset + mask.index_count),
            mask.vertex_offset as i32,
            0..1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn render_composite_blit_pooled(
        &self,
        render_context: &mut RenderContext,
        header: &CompositeHeader,
        framebuffer: &CompositeFramebufferEntry,
        parent_attachment: RenderPassColorAttachment,
        ctx_samples: u32,
        pool: &UniformPool,
        comp_idx: usize,
    ) {
        let Some(&pipeline_id) = self
            .pipeline_resource
            .composite_pipeline
            .get(&(header.blend_mode, ctx_samples))
        else {
            return;
        };
        let Some(cache_pipeline) = self.pipeline_cache.get_render_pipeline(pipeline_id) else {
            return;
        };

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("inx_composite_blit"),
            color_attachments: &[Some(parent_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(cache_pipeline);
        render_pass.set_vertex_buffer(0, self.composite_pool.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.composite_pool.uv_buffer.slice(..));
        render_pass.set_bind_group(0, &self.view_bindgroup.value, &[self.view_offset.offset]);

        // DYNAMIC OFFSET
        if let Some(ref comp_bg) = pool.composite_bind_group {
            render_pass.set_bind_group(1, comp_bg, &[pool.composite_offsets[comp_idx]]);
        }

        render_pass.set_bind_group(2, &framebuffer.bindgroup, &[]);
        render_pass.set_bind_group(3, &framebuffer.bindgroup, &[]);
        render_pass.set_bind_group(4, &framebuffer.bindgroup, &[]);
        render_pass.set_index_buffer(
            self.composite_pool.index_buffer.slice(..),
            IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..3, 0, 0..1);
    }
    /// Flush: dibuja todos los parts acumulados en UN solo render pass.
    fn flush_batch(
        &self,
        render_context: &mut RenderContext,
        batch: &[(usize, &InxPartData)],
        pool: &UniformPool,
        stack: &[CompositeFrame],
        stencil_ref: u32,
        composite_first_draw: &mut [bool],
    ) {
        if batch.is_empty() {
            return;
        }

        // Resolver target
        let (color_attachment, depth_stencil_view, ctx_samples) = if let Some(frame) = stack.last()
        {
            let first = composite_first_draw.last().copied().unwrap_or(false);
            if let Some(flag) = composite_first_draw.last_mut() {
                *flag = false;
            }

            let load = if first {
                LoadOp::Clear(LinearRgba::NONE.into())
            } else {
                LoadOp::Load
            };

            (
                RenderPassColorAttachment {
                    view: &frame.entry.albedo_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load,
                        store: StoreOp::Store,
                    },
                },
                &frame.entry.depth_stencil_view,
                1,
            )
        } else {
            (
                self.view_target.get_color_attachment(),
                &self.scene_buffer.depth_stencil_view,
                self.samples,
            )
        };

        // UN solo render pass para todo el batch
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("inx_batched_parts"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_stencil_view,
                depth_ops: None,
                stencil_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_stencil_reference(stencil_ref);

        // Set shared state ONCE
        render_pass.set_vertex_buffer(0, self.gpu_buffer.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.gpu_buffer.uv_buffer.slice(..));
        render_pass.set_vertex_buffer(2, self.gpu_buffer.deform_buffer.slice(..));
        render_pass.set_index_buffer(
            self.gpu_buffer.index_buffer.slice(..),
            IndexFormat::Uint32,
        );
        render_pass.set_bind_group(0, &self.view_bindgroup.value, &[self.view_offset.offset]);

        let mut current_blend: Option<BlendMode> = None;

        for &(part_idx, part) in batch {
            if current_blend != Some(part.blend_mode)
                && let Some(&pid) = self
                    .pipeline_resource
                    .basic_pipeline
                    .get(&(part.blend_mode, ctx_samples))
                && let Some(pipe) = self.pipeline_cache.get_render_pipeline(pid)
            {
                render_pass.set_render_pipeline(pipe);
                current_blend = Some(part.blend_mode);
            }

            // dynamic offset
            let offset = pool.part_offsets[part_idx];
            render_pass.set_bind_group(1, &pool.part_bind_group, &[offset]);

            self.bind_texture(self.puppet_entity, &mut render_pass, 2, part.textures[0]);
            self.bind_texture(self.puppet_entity, &mut render_pass, 3, part.textures[1]);
            self.bind_texture(self.puppet_entity, &mut render_pass, 4, part.textures[2]);

            render_pass.draw_indexed(
                part.index_offset..(part.index_offset + part.index_count),
                part.vertex_offset as i32,
                0..1,
            );
        }
    }

    // Agregar a impl InxRenderPass:
    fn build_uniform_pool(&self, data: &InxData) -> UniformPool {
        let min_align = self
            .render_device
            .limits()
            .min_uniform_buffer_offset_alignment;

        // Parts + Masks: usan InxUniform
        let inx_uniform_size = InxUniform::min_size().get() as u32;
        let inx_aligned = align_up(inx_uniform_size, min_align);

        // Contar parts y masks
        let mut part_uniforms: Vec<InxUniform> = Vec::new();
        let mut mask_uniforms: Vec<InxUniform> = Vec::new();
        let mut composite_uniforms: Vec<CompositeUniform> = Vec::new();

        for cmd in &data.commands {
            match cmd {
                RenderOrder::DrawPart(part) => {
                    part_uniforms.push(InxUniform::new(
                        part.transform,
                        part.origin,
                        part.tint,
                        part.screen_tint,
                        part.opacity,
                        part.emissive_strength,
                        part.mask_threshold,
                    ));
                }
                RenderOrder::PushMask(mask) => {
                    mask_uniforms.push(InxUniform::new(
                        mask.transform,
                        mask.origin,
                        Vec3::ONE,
                        Vec3::ZERO,
                        1.0,
                        0.0,
                        mask.threshold,
                    ));
                }
                RenderOrder::BeginComposite(header) => {
                    composite_uniforms.push(CompositeUniform::new(
                        header.transform,
                        header.opacity,
                        header.tint,
                        header.screen_tint,
                    ));
                }
                _ => {}
            }
        }

        // Build part buffer
        let part_bind_group;
        let part_offsets;

        if part_uniforms.is_empty() {
            // Necesitamos al menos un dummy para el bind group
            let dummy_buf = self.render_device.create_buffer(&BufferDescriptor {
                label: Some("inx_uniform_pool_dummy"),
                size: inx_aligned as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            part_bind_group = self.create_dynamic_bind_group(
                &dummy_buf,
                &self.pipeline_resource.basic_uniform_layout,
                InxUniform::min_size(),
            );
            part_offsets = Vec::new();
        } else {
            let (buf, offsets) = self.write_uniform_buffer(&part_uniforms, inx_aligned);
            part_bind_group = self.create_dynamic_bind_group(
                &buf,
                &self.pipeline_resource.basic_uniform_layout,
                InxUniform::min_size(),
            );
            part_offsets = offsets;
        }

        // Build mask buffer (comparte layout con parts)
        let (mask_bind_group, mask_offsets) = if mask_uniforms.is_empty() {
            (None, Vec::new())
        } else {
            let (buf, offsets) = self.write_uniform_buffer(&mask_uniforms, inx_aligned);
            let bg = self.create_dynamic_bind_group(
                &buf,
                &self.pipeline_resource.basic_uniform_layout,
                InxUniform::min_size(),
            );
            (Some(bg), offsets)
        };

        // Build composite buffer
        let comp_uniform_size = CompositeUniform::min_size().get() as u32;
        let comp_aligned = align_up(comp_uniform_size, min_align);

        let (composite_bind_group, composite_offsets) = if composite_uniforms.is_empty() {
            (None, Vec::new())
        } else {
            let (buf, offsets) = self.write_uniform_buffer(&composite_uniforms, comp_aligned);
            let bg = self.create_dynamic_bind_group(
                &buf,
                &self.pipeline_resource.composite_uniform_layout,
                CompositeUniform::min_size(),
            );
            (Some(bg), offsets)
        };

        UniformPool {
            part_bind_group,
            part_offsets,
            composite_bind_group,
            composite_offsets,
            mask_bind_group,
            mask_offsets,
        }
    }

    /// Escribe un Vec de uniforms T en un buffer alineado.
    /// Retorna (Buffer, Vec<offset_in_bytes>).
    fn write_uniform_buffer<T: ShaderType + WriteInto>(
        &self,
        uniforms: &[T],
        aligned_size: u32,
    ) -> (Buffer, Vec<u32>) {
        let total_size = aligned_size as u64 * uniforms.len() as u64;

        // CPU-side: serializar cada uniform en su slot alineado
        let mut cpu_data = vec![0u8; total_size as usize];
        let mut offsets = Vec::with_capacity(uniforms.len());

        for (i, uniform) in uniforms.iter().enumerate() {
            let offset = i as u32 * aligned_size;
            offsets.push(offset);

            let mut writer = UniformBuffer::new(Vec::new());
            if writer.write(uniform).is_ok() {
                let bytes = writer.as_ref();
                let start = offset as usize;
                let end = start + bytes.len().min(aligned_size as usize);
                cpu_data[start..end].copy_from_slice(&bytes[..end - start]);
            }
        }

        let buffer = self
            .render_device
            .create_buffer_with_data(&BufferInitDescriptor {
                label: Some("inx_uniform_pool"),
                contents: &cpu_data,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        (buffer, offsets)
    }

    fn create_dynamic_bind_group(
        &self,
        buffer: &Buffer,
        layout: &BindGroupLayout,
        min_binding_size: std::num::NonZeroU64,
    ) -> BindGroup {
        self.render_device.create_bind_group(
            Some("inx_dynamic_uniform_bg"),
            layout,
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer,
                    offset: 0,
                    size: Some(min_binding_size),
                }),
            }],
        )
    }

    fn bind_texture(
        &self,
        puppet_entity: Entity,
        render_pass: &mut TrackedRenderPass<'r>,
        group: usize,
        texture_idx: u32,
    ) {
        let fallback = self.textures.fallback.as_ref().unwrap();

        if texture_idx == u32::MAX {
            render_pass.set_bind_group(group, fallback, &[]);
            return;
        }

        // Resolver: local index - AssetId - shared BindGroup
        let bind_group = self
            .textures
            .entity_maps
            .get(&puppet_entity)
            .and_then(|map| map.get(texture_idx as usize))
            .and_then(|asset_id| self.textures.by_asset.get(asset_id))
            .unwrap_or(fallback);

        render_pass.set_bind_group(group, bind_group, &[]);
    }
}

#[inline]
fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}
