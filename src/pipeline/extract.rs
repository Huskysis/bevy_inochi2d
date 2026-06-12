use bevy::{
    camera::visibility::RenderLayers,
    platform::collections::HashMap,
    prelude::*,
    render::{
        Extract,
        sync_world::RenderEntity,
    },
};

use crate::{
    BlendMode, InxDeform, InxMaskMode, InxMaterial, InxNodeType, InxStructureVersion, InxUUID,
    InxZSort,
};

use super::*;

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
    /// Layers this puppet belongs to; views only draw intersecting puppets.
    pub layers: RenderLayers,
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
            (
                Entity,
                &RenderEntity,
                Option<&InxStructureVersion>,
                Option<&RenderLayers>,
            ),
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
    for (_entity, render_entity, version, _layers) in roots_render.iter() {
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

        for (entity, render_entity, version, layers) in roots_render.iter() {
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
                    data.layers = layers.cloned().unwrap_or_default();
                }
            } else {
                // Nuevo puppet o estructura cambiada - full (re)build
                let uuid_map = build_uuid_map_for_root(root, &node_map);
                let mut data = InxData {
                    structure_version: current_version,
                    layers: layers.cloned().unwrap_or_default(),
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

        for (_entity, render_entity, _version, layers) in roots_render.iter() {
            if let Ok(mut data) = existing_data.get_mut(render_entity.id()) {
                update_dynamic_data_light(&mut data, &light_map);
                data.layers = layers.cloned().unwrap_or_default();
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

