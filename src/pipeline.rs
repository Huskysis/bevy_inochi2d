use bevy::{
    asset::load_embedded_asset,
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    mesh::VertexBufferLayout,
    platform::collections::HashMap,
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
        render_asset::RenderAssets,
        render_graph::{RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner},
        render_phase::TrackedRenderPass,
        render_resource::{
            encase::{UniformBuffer, private::WriteInto},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_world::RenderEntity,
        texture::{FallbackImage, GpuImage},
        view::{ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
    shader::load_shader_library,
};

use crate::{
    BlendMode, InxDeform, InxMaskMode, InxMaterial, InxNodeType, InxStructureVersion, InxUUID,
    InxZSort,
};

/// Mapa estatico: para cada comando que tiene datos dinamicos,
/// guarda que Entity del main world leer para actualizar.
/// Se construye una vez y vive en el render world.
#[derive(Component)]
pub struct InxExtractMap {
    /// Para cada comando en InxData.commands, si es DrawPart:
    ///
    ///   Some((main_entity, deform_start, deform_count))
    ///
    /// Si es otro tipo: None
    pub entries: Vec<ExtractMapEntry>,
}

pub enum ExtractMapEntry {
    Part {
        main_entity: Entity,
        deform_start: usize,
        deform_count: usize,
        command_idx: usize,
    },
    Mask {
        main_entity: Entity,
        command_idx: usize,
    },
    Composite {
        main_entity: Entity,
        command_idx: usize,
    },
    None,
}

#[derive(Debug, Default, Component)]
pub struct InxData {
    pub verts: Vec<[f32; 2]>,
    pub uvs: Vec<[f32; 2]>,
    pub deforms: Vec<[f32; 2]>,
    pub deform_dirty: Option<(u32, u32)>, // (start_byte, end_byte)
    pub indices: Vec<u32>,
    pub commands: Vec<RenderOrder>,
    pub textures: Vec<AssetId<Image>>,
    /// Copy of the root's `InxStructureVersion` at build time; mismatch
    /// triggers a full command-list rebuild (props added/removed).
    pub structure_version: u32,
}

#[derive(Debug, Clone)]
pub enum RenderOrder {
    DrawPart(InxPartData),
    BeginComposite(CompositeHeader),
    EndComposite,
    PushMask(MaskHeader),
    PopMask,
}

impl RenderOrder {
    pub fn name(&self) -> String {
        match self {
            RenderOrder::DrawPart(part) => part.name.clone(),
            RenderOrder::BeginComposite(header) => header.name.clone(),
            RenderOrder::EndComposite => "End Composite".to_string(),
            RenderOrder::PushMask(header) => header.name.clone(),
            RenderOrder::PopMask => "Pop Mask".to_string(),
        }
    }
    pub fn z_sort(&self) -> f32 {
        match self {
            RenderOrder::DrawPart(part) => part.transform.w_axis.z,
            RenderOrder::BeginComposite(header) => header.transform.w_axis.z,
            RenderOrder::EndComposite => 0.0,
            RenderOrder::PushMask(_header) => 0.0,
            RenderOrder::PopMask => 0.0,
        }
    }
    pub fn is_part(&self) -> bool {
        matches!(self, Self::DrawPart(_))
    }
    pub fn is_composite(&self) -> bool {
        matches!(self, Self::BeginComposite(_))
    }
}

#[derive(Debug, Clone)]
pub struct InxPartData {
    pub entity: Entity,
    pub uuid: u32,
    pub name: String,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub deform_start: u32,
    pub textures: [u32; 3],
    pub tint: Vec3,
    pub screen_tint: Vec3,
    pub emissive_strength: f32,
    pub blend_mode: BlendMode,
    pub mask_threshold: f32,
    pub opacity: f32,
    pub transform: Mat4,
    pub origin: Vec2,
}

#[derive(Debug, Clone)]
pub struct CompositeHeader {
    pub entity: Entity,
    pub uuid: u32,
    pub name: String,
    pub tint: Vec3,
    pub screen_tint: Vec3,
    pub blend_mode: BlendMode,
    pub opacity: f32,
    pub transform: Mat4,
}

#[derive(Debug, Clone)]
pub struct MaskHeader {
    pub entity: Entity,
    pub entity_source: Entity,
    pub source_uuid: u32,
    pub mode: MaskMode,
    pub threshold: f32,

    // Geometria del source node
    pub name: String,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub tex_albedo: u32,
    pub transform: Mat4,
    pub origin: Vec2,
}

#[derive(Debug, Clone, Default)]
pub enum MaskMode {
    #[default]
    Mask,
    Dodge,
}

enum SortableNode {
    Part {
        data: InxPartData,
        masks: Vec<MaskHeader>,
        zsort: f32,
    },
    Composite {
        header: CompositeHeader,
        children: Vec<SortableNode>,
        zsort: f32,
    },
}

impl SortableNode {
    fn zsort(&self) -> f32 {
        match self {
            SortableNode::Part { zsort, .. } => *zsort,
            SortableNode::Composite { zsort, .. } => *zsort,
        }
    }
}

#[derive(Debug)]
struct ExtractedNode {
    entity: Entity,
    uuid: u32,
    name: String,
    node_type: InxNodeType,
    zsort: f32,
    material: Option<InxMaterial>,
    deform: Option<InxDeform>,
    global_transform: Mat4,
    children: Vec<Entity>,
}

pub fn extract_inx_node(
    mut commands: Commands,

    // Main world (Extract<>)
    roots: Extract<Query<Entity, (With<InxUUID>, Without<ChildOf>)>>,
    roots_render: Extract<
        Query<
            (Entity, &RenderEntity, Option<&InxStructureVersion>),
            (With<InxUUID>, Without<ChildOf>),
        >,
    >,

    // Full query - solo se usa en slow path
    nodes_full: Extract<
        Query<(
            Entity,
            &InxUUID,
            &InxZSort,
            &InxNodeType,
            &ViewVisibility,
            &GlobalTransform,
            Option<&InxMaterial>,
            Option<&Children>,
            Option<&Name>,
            Option<&InxDeform>,
        )>,
    >,

    // Light query - solo lo que cambia cada frame (fast path)
    nodes_light: Extract<
        Query<(
            Entity,
            &InxUUID,
            &GlobalTransform,
            Option<&InxMaterial>,
            Option<&InxDeform>,
        )>,
    >,

    // Render world (sin Extract)
    mut existing_data: Query<&mut InxData>,
    mut tex_bind: ResMut<InxTexturesBindGroup>,
) {
    // Primero: detectar si ALGUN puppet necesita slow path
    // (sin InxData, o la estructura cambio — props añadidos/quitados)
    let mut needs_slow_path = false;
    for (_entity, render_entity, version) in roots_render.iter() {
        match existing_data.get(render_entity.id()) {
            Err(_) => {
                needs_slow_path = true;
                break;
            }
            Ok(data) => {
                if data.structure_version != version.map(|v| v.0).unwrap_or(0) {
                    needs_slow_path = true;
                    break;
                }
            }
        }
    }

    if needs_slow_path {
        // SLOW PATH: al menos un puppet es nuevo
        // Construir node_map completo (con material clone)
        let mut node_map: HashMap<Entity, ExtractedNode> = HashMap::new();

        for (entity, uuid, zsort, node_type, view, gtransform, material, children, name, deform) in
            nodes_full.iter()
        {
            if !view.get() {
                continue;
            }
            node_map.insert(
                entity,
                ExtractedNode {
                    entity,
                    uuid: uuid.0,
                    name: name.map(|n| n.to_string()).unwrap_or_default(),
                    zsort: zsort.0,
                    node_type: *node_type,
                    material: material.cloned(),
                    deform: deform.cloned(),
                    global_transform: gtransform.to_matrix(),
                    children: children.map(|c| c.iter().collect()).unwrap_or_default(),
                },
            );
        }

        for (entity, render_entity, version) in roots_render.iter() {
            let Ok(root) = roots.get(entity) else {
                continue;
            };

            let current_version = version.map(|v| v.0).unwrap_or(0);
            let up_to_date = existing_data
                .get(render_entity.id())
                .is_ok_and(|data| data.structure_version == current_version);

            if up_to_date {
                if let Ok(mut data) = existing_data.get_mut(render_entity.id()) {
                    update_dynamic_data(&mut data, &node_map);
                }
            } else {
                // Nuevo puppet o estructura cambiada - full (re)build
                let uuid_map = build_uuid_map_for_root(root, &node_map);
                let mut data = InxData {
                    structure_version: current_version,
                    ..Default::default()
                };
                let mut tex_lookup: HashMap<AssetId<Image>, u32> = HashMap::new();

                let mut sortable =
                    collect_subtree(root, &node_map, &uuid_map, &mut data, &mut tex_lookup, 0.0);
                sort_nodes(&mut sortable);
                flatten_to_commands(sortable, &mut data.commands);

                // Invalidar caches por-entity: GPU buffers y mapa de texturas
                // se reconstruyen en prepare a partir del nuevo InxData.
                tex_bind.entity_maps.remove(&render_entity.id());
                commands
                    .entity(render_entity.id())
                    .remove::<PuppetGpuBuffers>()
                    .insert(data);
            }
        }
    } else {
        // FAST PATH: todos los puppets ya existen
        // Construir un mapa ligero: uuid con transform, opacity, deform
        // SIN clonar InxMaterial.mesh

        let mut light_map: HashMap<(Entity, u32), LightNode> = HashMap::default();

        for (entity, uuid, gtransform, material, deform) in nodes_light.iter() {
            light_map.insert(
                (entity, uuid.0),
                LightNode {
                    transform: gtransform.to_matrix(),
                    opacity: material.map(|m| m.opacity).unwrap_or(1.0),
                    deform_offsets: deform.map(|d| d.offsets.as_slice()),
                    tint: material.map(|m| m.tint).unwrap_or(Vec3::ONE),
                    screen_tint: material.map(|m| m.screen_tint).unwrap_or(Vec3::ZERO),
                    emissive_strength: material.map(|m| m.emissive_strength).unwrap_or(0.0),
                    blend_mode: material.map(|m| m.blend_mode).unwrap_or_default(),
                    mask_threshold: material.map(|m| m.mask_threshold).unwrap_or(0.5),
                },
            );
        }

        for (_entity, render_entity, _version) in roots_render.iter() {
            if let Ok(mut data) = existing_data.get_mut(render_entity.id()) {
                update_dynamic_data_light(&mut data, &light_map);
            }
        }
    }
}

fn build_uuid_map_for_root(
    root: Entity,
    node_map: &HashMap<Entity, ExtractedNode>,
) -> HashMap<u32, Entity> {
    let mut map = HashMap::new();
    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        if let Some(node) = node_map.get(&entity) {
            map.insert(node.uuid, entity);
            stack.extend_from_slice(&node.children);
        }
    }
    map
}

struct LightNode<'a> {
    transform: Mat4,
    opacity: f32,
    deform_offsets: Option<&'a [[f32; 2]]>,
    tint: Vec3,
    screen_tint: Vec3,
    emissive_strength: f32,
    blend_mode: BlendMode,
    mask_threshold: f32,
    // TODO!: añadir mas datos para actualizar cuando cambian
}

/// Fast path: actualiza con el mapa ligero (sin material clone).
fn update_dynamic_data_light(data: &mut InxData, light_map: &HashMap<(Entity, u32), LightNode>) {
    for cmd in data.commands.iter_mut() {
        match cmd {
            RenderOrder::DrawPart(part) => {
                if let Some(node) = light_map.get(&(part.entity, part.uuid)) {
                    part.transform = node.transform;
                    part.opacity = node.opacity;
                    part.tint = node.tint;
                    part.screen_tint = node.screen_tint;
                    part.emissive_strength = node.emissive_strength;
                    part.blend_mode = node.blend_mode;
                    part.mask_threshold = node.mask_threshold;

                    if let Some(offsets) = node.deform_offsets {
                        let start = part.deform_start as usize;
                        let count = offsets.len();
                        if start + count <= data.deforms.len() {
                            data.deforms[start..start + count].copy_from_slice(offsets);

                            // Expandir el rango de actualizacion
                            let byte_start = (start * std::mem::size_of::<[f32; 2]>()) as u32;
                            let byte_end =
                                ((start + count) * std::mem::size_of::<[f32; 2]>()) as u32;
                            data.deform_dirty = Some(match data.deform_dirty {
                                Some((s, e)) => (s.min(byte_start), e.max(byte_end)),
                                None => (byte_start, byte_end),
                            });
                        }
                    }
                }
            }

            RenderOrder::PushMask(mask) => {
                if let Some(node) = light_map.get(&(mask.entity_source, mask.source_uuid)) {
                    mask.transform = node.transform;
                }
            }

            RenderOrder::BeginComposite(header) => {
                if let Some(node) = light_map.get(&(header.entity, header.uuid)) {
                    header.transform = node.transform;
                    header.opacity = node.opacity;
                    header.tint = node.tint;
                    header.screen_tint = node.screen_tint;
                    header.blend_mode = node.blend_mode;
                }
            }

            _ => {}
        }
    }
}

/// Slow path fallback: actualiza con el node_map completo.
fn update_dynamic_data(data: &mut InxData, node_map: &HashMap<Entity, ExtractedNode>) {
    for cmd in data.commands.iter_mut() {
        match cmd {
            RenderOrder::DrawPart(part) => {
                if let Some(node) = node_map.get(&part.entity) {
                    part.transform = node.global_transform;
                    if let Some(mat) = &node.material {
                        part.opacity = mat.opacity;
                    }
                    if let Some(deform) = &node.deform {
                        let start = part.deform_start as usize;
                        let count = deform.offsets.len();
                        if start + count <= data.deforms.len() {
                            data.deforms[start..start + count].copy_from_slice(&deform.offsets);
                        }
                    }
                }
            }

            RenderOrder::PushMask(mask) => {
                if let Some(node) = node_map.get(&mask.entity_source) {
                    mask.transform = node.global_transform;
                }
            }

            RenderOrder::BeginComposite(header) => {
                if let Some(node) = node_map.get(&header.entity) {
                    header.transform = node.global_transform;
                    if let Some(mat) = &node.material {
                        header.opacity = mat.opacity;
                    }
                }
            }

            _ => {}
        }
    }

    if !data.deforms.is_empty() {
        data.deform_dirty = Some((0, (data.deforms.len() * size_of::<[f32; 2]>()) as u32));
    }
}

fn collect_subtree(
    entity: Entity,
    nodes: &HashMap<Entity, ExtractedNode>,
    uuid_map: &HashMap<u32, Entity>,
    data: &mut InxData,
    tex_lookup: &mut HashMap<AssetId<Image>, u32>,
    parent_zsort: f32,
) -> Vec<SortableNode> {
    let Some(node) = nodes.get(&entity) else {
        return vec![];
    };
    let accumulated = parent_zsort + node.zsort;

    match node.node_type {
        InxNodeType::Part => collect_part(node, nodes, uuid_map, data, tex_lookup, accumulated),
        InxNodeType::Composite => {
            collect_composite(node, nodes, uuid_map, data, tex_lookup, accumulated)
        }
        InxNodeType::Mask => vec![], // consumido por el padre
        _ => collect_children_flat(node, nodes, uuid_map, data, tex_lookup, accumulated),
    }
}

fn collect_part(
    node: &ExtractedNode,
    nodes: &HashMap<Entity, ExtractedNode>,
    uuid_map: &HashMap<u32, Entity>,
    data: &mut InxData,
    tex_lookup: &mut HashMap<AssetId<Image>, u32>,
    accumulated: f32,
) -> Vec<SortableNode> {
    let Some(mat) = &node.material else {
        return collect_children_flat(node, nodes, uuid_map, data, tex_lookup, accumulated);
    };
    let Some(mesh) = &mat.mesh else {
        return collect_children_flat(node, nodes, uuid_map, data, tex_lookup, accumulated);
    };

    // Geometria flat buffers
    let vertex_offset = data.verts.len() as u32;
    let vertex_count = mesh.vertex_buffer.len() as u32;
    data.verts.extend_from_slice(&mesh.vertex_buffer);
    data.uvs.extend_from_slice(&mesh.uv_buffer);

    let deform_start = data.deforms.len() as u32;

    if let Some(deform_offsets) = &node.deform {
        data.deforms.extend_from_slice(&deform_offsets.offsets);
    } else {
        // Sin deformacion, zeros
        // Debe tener el mismo largo que verts
        data.deforms
            .extend(std::iter::repeat_n([0.0f32, 0.0], vertex_count as usize));
    }

    let index_offset = data.indices.len() as u32;
    let index_count = mesh.index_buffer.len() as u32;
    data.indices
        .extend(mesh.index_buffer.iter().map(|&i| i + vertex_offset));

    let textures = resolve_textures(mat, &mut data.textures, tex_lookup);
    let masks = collect_masks(node, nodes, uuid_map, data, tex_lookup);

    let part_data = InxPartData {
        entity: node.entity,
        uuid: node.uuid,
        name: node.name.clone(),
        vertex_offset: if index_count > 0 { 0 } else { vertex_offset },
        vertex_count,
        index_offset,
        index_count,
        deform_start,
        textures,
        tint: mat.tint,
        screen_tint: mat.screen_tint,
        emissive_strength: mat.emissive_strength,
        blend_mode: mat.blend_mode,
        mask_threshold: mat.mask_threshold,
        opacity: mat.opacity,
        transform: node.global_transform,
        origin: mesh.origin,
    };

    // Este Part como nodo sorteable
    let mut result = vec![SortableNode::Part {
        data: part_data,
        masks,
        zsort: accumulated,
    }];

    // Hijos del Part van al MISMO nivel (no anidados dentro del Part)
    // para que participen en el sort global
    result.extend(collect_children_flat(
        node,
        nodes,
        uuid_map,
        data,
        tex_lookup,
        accumulated,
    ));
    result
}

fn collect_composite(
    node: &ExtractedNode,
    nodes: &HashMap<Entity, ExtractedNode>,
    uuid_map: &HashMap<u32, Entity>,
    data: &mut InxData,
    tex_lookup: &mut HashMap<AssetId<Image>, u32>,
    accumulated: f32,
) -> Vec<SortableNode> {
    let mat = node.material.as_ref();

    // Hijos del composite se sortean INTERNAMENTE (no escapan al nivel global)
    let children = collect_children_flat(node, nodes, uuid_map, data, tex_lookup, accumulated);

    let header = CompositeHeader {
        entity: node.entity,
        uuid: node.uuid,
        name: node.name.clone(),
        tint: mat.map(|m| m.tint).unwrap_or(Vec3::ONE),
        screen_tint: mat.map(|m| m.screen_tint).unwrap_or(Vec3::ZERO),
        blend_mode: mat.map(|m| m.blend_mode).unwrap_or_default(),
        opacity: mat.map(|m| m.opacity).unwrap_or(1.0),
        transform: node.global_transform,
    };

    // Composite = UNA unidad en sort global, con sus hijos adentro
    vec![SortableNode::Composite {
        header,
        children,
        zsort: accumulated,
    }]
}

fn collect_masks(
    node: &ExtractedNode,
    nodes: &HashMap<Entity, ExtractedNode>,
    uuid_map: &HashMap<u32, Entity>,
    data: &mut InxData,
    tex_lookup: &mut HashMap<AssetId<Image>, u32>,
) -> Vec<MaskHeader> {
    let Some(mat) = &node.material else {
        return vec![];
    };
    if mat.masks.is_empty() {
        return vec![];
    };

    let mut masks = Vec::new();
    for inx_mask in &mat.masks {
        // Encontrar el nodo fuente por UUID
        let Some(&source_entity) = uuid_map.get(&inx_mask.source_uuid) else {
            continue;
        };
        let Some(source_node) = nodes.get(&source_entity) else {
            continue;
        };
        let Some(source_mat) = &source_node.material else {
            continue;
        };
        let Some(source_mesh) = &source_mat.mesh else {
            continue;
        };

        // Geometria del source, osea flat buffers
        let vertex_offset = data.verts.len() as u32;
        let vertex_count = source_mesh.vertex_buffer.len() as u32;
        data.verts.extend_from_slice(&source_mesh.vertex_buffer);
        data.uvs.extend_from_slice(&source_mesh.uv_buffer);
        data.deforms.resize(data.verts.len(), [0.0, 0.0]);

        let index_offset = data.indices.len() as u32;
        let index_count = source_mesh.index_buffer.len() as u32;
        data.indices
            .extend(source_mesh.index_buffer.iter().map(|&i| i + vertex_offset));

        let tex_albedo = resolve_textures(source_mat, &mut data.textures, tex_lookup)[0];

        masks.push(MaskHeader {
            entity: node.entity,
            entity_source: source_node.entity,
            source_uuid: inx_mask.source_uuid,
            mode: match inx_mask.mode {
                InxMaskMode::Dodge => MaskMode::Dodge,
                _ => MaskMode::Mask,
            },
            threshold: mat.mask_threshold,

            name: source_node.name.clone(),
            vertex_offset: if index_count > 0 { 0 } else { vertex_offset },
            vertex_count,
            index_offset,
            index_count,
            tex_albedo,
            transform: source_node.global_transform,
            origin: source_mesh.origin,
        });
    }
    masks
}

/// Recolecta hijos, aplanando Generic/MeshGroup/entre otros
fn collect_children_flat(
    node: &ExtractedNode,
    nodes: &HashMap<Entity, ExtractedNode>,
    uuid_map: &HashMap<u32, Entity>,
    data: &mut InxData,
    tex_lookup: &mut HashMap<AssetId<Image>, u32>,
    parent_accumulated: f32,
) -> Vec<SortableNode> {
    let mut result = Vec::new();
    for &child in &node.children {
        result.extend(collect_subtree(
            child,
            nodes,
            uuid_map,
            data,
            tex_lookup,
            parent_accumulated,
        ));
    }
    result
}

/// Sort recursivo: cada nivel se ordena por zsort descendente
/// (mayor zsort = mas atras = dibuja primero)
fn sort_nodes(nodes: &mut [SortableNode]) {
    nodes.sort_by(|a, b| {
        b.zsort()
            .partial_cmp(&a.zsort())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for node in nodes.iter_mut() {
        if let SortableNode::Composite { children, .. } = node {
            sort_nodes(children);
        }
    }
}

fn flatten_to_commands(nodes: Vec<SortableNode>, commands: &mut Vec<RenderOrder>) {
    for node in nodes {
        match node {
            SortableNode::Part { data, masks, .. } => {
                for mask in &masks {
                    commands.push(RenderOrder::PushMask(mask.clone()));
                }
                commands.push(RenderOrder::DrawPart(data));
                for _ in &masks {
                    commands.push(RenderOrder::PopMask);
                }
            }
            SortableNode::Composite {
                header, children, ..
            } => {
                commands.push(RenderOrder::BeginComposite(header));
                flatten_to_commands(children, commands);
                commands.push(RenderOrder::EndComposite);
            }
        }
    }
}

const TEX_NONE: u32 = u32::MAX;

fn resolve_textures(
    mat: &InxMaterial,
    tex_map: &mut Vec<AssetId<Image>>,
    tex_lookup: &mut HashMap<AssetId<Image>, u32>,
) -> [u32; 3] {
    let mut resolve = |handle: &Option<Handle<Image>>| -> u32 {
        match handle {
            Some(h) => {
                let id = h.id();
                *tex_lookup.entry(id).or_insert_with(|| {
                    let idx = tex_map.len() as u32;
                    tex_map.push(id);
                    idx
                })
            }
            None => TEX_NONE,
        }
    };

    [
        resolve(&mat.texture_albedo),
        resolve(&mat.texture_emissive),
        resolve(&mat.texture_bumpmap),
    ]
}

pub struct InxRenderPipeline;

impl bevy::app::Plugin for InxRenderPipeline {
    fn build(&self, app: &mut bevy::prelude::App) {
        load_shader_library!(app, "basic.wgsl");
        load_shader_library!(app, "mask.wgsl");
        load_shader_library!(app, "composite.wgsl");
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
    value: BindGroup,
}

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

#[derive(Resource)]
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
#[derive(Resource)]
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

        let shader_basic = load_embedded_asset!(assets, "basic.wgsl");
        let shader_mask = load_embedded_asset!(assets, "mask.wgsl");
        let shader_composite = load_embedded_asset!(assets, "composite.wgsl");

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
    views: Query<&ViewTarget>,
    mut pipeline: ResMut<InxPipeline>,
    composite: Option<ResMut<CompositeFramebufferPool>>,
    scene: Option<ResMut<SceneFramebuffer>>,
) {
    let Some(view) = views.iter().next() else {
        return;
    };

    let size = view.main_texture().size();
    let viewport_size = UVec2::new(size.width, size.height);
    // main_texture() is the resolve target (1x); MSAA texture is separate
    let samples = view
        .sampled_main_texture()
        .map(|t| t.sample_count())
        .unwrap_or(1);

    pipeline.ensure_samples(samples, &pipeline_cache);

    if let Some(mut framebuffer) = composite {
        framebuffer.resize(&render_device, viewport_size, &pipeline);
    } else {
        commands.insert_resource(CompositeFramebufferPool::new(
            &render_device,
            viewport_size,
            &pipeline,
        ));
    }

    if let Some(mut fb) = scene {
        fb.resize(&render_device, viewport_size, samples);
    } else {
        commands.insert_resource(SceneFramebuffer::new(
            &render_device,
            viewport_size,
            samples,
        ));
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
    type ViewQuery = &'static ViewTarget;

    fn update(&mut self, world: &mut World) {
        self.extract_buffer.update_archetypes(world);
        self.view_bindgroup.update_archetypes(world);
        self.view_offset.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext<'w>,
        view_target: bevy::ecs::query::QueryItem<'w, '_, Self::ViewQuery>,
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

        let Some(composite_fb_pool) = world.get_resource::<CompositeFramebufferPool>() else {
            return Ok(());
        };

        let Some(scene_fb) = world.get_resource::<SceneFramebuffer>() else {
            return Ok(());
        };

        let render_device = world.resource::<RenderDevice>();

        let samples = view_target
            .sampled_main_texture()
            .map(|t| t.sample_count())
            .unwrap_or(1);

        // Obtener puppets
        for (entity, data, puppet_gpu_buffers) in self.extract_buffer.iter_manual(world) {
            if data.commands.is_empty() {
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
