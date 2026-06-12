use bevy::{
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
        render_graph::{RenderGraphExt, ViewNodeRunner},
    },
    shader::load_shader_library,
};

use crate::InxUUID;


pub mod extract;
pub mod framebuffers;
pub mod node;
pub mod prepare;

pub use extract::*;
pub use framebuffers::*;
pub use node::*;
pub use prepare::*;

pub struct InxRenderPipeline;

impl bevy::app::Plugin for InxRenderPipeline {
    fn build(&self, app: &mut bevy::prelude::App) {
        load_shader_library!(app, "../basic.wgsl");
        load_shader_library!(app, "../mask.wgsl");
        load_shader_library!(app, "../composite.wgsl");
        app.add_plugins(ExtractComponentPlugin::<InxUUID>::default())
            .add_plugins(UniformComponentPlugin::<InxUniform>::default());

        let render_app = app.sub_app_mut(RenderApp);

        render_app
            .init_resource::<InxTexturesBindGroup>()
            .add_systems(ExtractSchedule, extract_inx_node)
            .add_systems(
                Render,
                (
                    prepare_puppet_buffers.in_set(RenderSystems::Prepare),
                    prepare_view_target_composite_scene.in_set(RenderSystems::PrepareResources),
                    prepare_texture_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    prepare_inx_view_bind_group.in_set(RenderSystems::PrepareBindGroups),
                    update_deform_buffer
                        .in_set(RenderSystems::PrepareResources)
                        .after(prepare_puppet_buffers),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<InxRenderViewNode>>(Core2d, InxViewNodeLabel)
            .add_render_graph_edges(Core2d, (Node2d::MainOpaquePass, InxViewNodeLabel));
    }
    fn finish(&self, _app: &mut App) {
        let Some(render_app) = _app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<InxPipeline>();
    }
}

