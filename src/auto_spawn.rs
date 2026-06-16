//! Consumes [`InxScene`] command-components and spawns the puppet tree.
//!
//! Walks the loaded [`InxPuppet`] node hierarchy once the asset is ready,
//! materializing one ECS entity per node with its transform, material,
//! base pose, deform buffer and (when requested) animation controller,
//! then propagates the parent's [`RenderLayers`] to every child.

use bevy::{camera::visibility::RenderLayers, prelude::*};

use crate::{
    InxAnimationController, InxBasePose, InxDeform, InxNode, InxNodeType, InxParam, InxParamState,
    InxPuppet, InxPuppetRoot, InxScene, InxUUID, InxZSort,
    simple_physics::{InxPhysicsState, InxSimplePhysics},
};

/// Sistema que procesa los comandos InxScene y spawnea el árbol de nodos.
pub fn spawn_scene_system(
    mut commands: Commands,
    query: Query<(Entity, &InxScene, Option<&RenderLayers>)>,
    puppets: Res<Assets<InxPuppet>>,
    param_assets: Res<Assets<InxParam>>,
) {
    for (entity, scene, layers) in query.iter() {
        let Some(puppet) = puppets.get(&scene.puppet) else {
            continue; // Asset aun no cargado
        };

        // Remover el comando para no re-procesar
        commands.entity(entity).remove::<InxScene>();

        // Buscar nodo raiz (ultimo nodo = raiz del árbol)
        let Some(root_node) = puppet.nodes.last() else {
            eprintln!("InxScene: puppet sin nodos");
            commands.entity(entity).despawn();
            continue;
        };

        // Spawnear árbol recursivamente
        let root_entity = spawn_node_recursive(&mut commands, root_node, scene.transform);

        // Marcar raiz con componentes de puppet
        commands.entity(root_entity).insert((
            InxPuppetRoot {
                source: scene.puppet.clone(),
            },
            scene.transform,
        ));

        // Propagate RenderLayers from the command entity to the puppet root
        // (the command entity is despawned below).
        if let Some(layers) = layers {
            commands.entity(root_entity).insert(layers.clone());
        }

        // Param state
        let param_state = init_param_state(puppet, &param_assets);
        commands.entity(root_entity).insert(param_state);

        // Animacion (si aplica)
        if scene.animation {
            let mut controller = InxAnimationController::new();

            // Defaults de params
            for param_handle in &puppet.params {
                if let Some(param) = param_assets.get(param_handle) {
                    controller.param_defaults.insert(param.uuid, param.defaults);
                }
            }

            // Todas las animaciones se registran pero NO se reproducen automáticamente.
            // El usuario decide que reproducir despues con controller.play() / set_idle().
            commands.entity(root_entity).insert(controller);
        }

        // Eliminar entidad temporal del comando
        commands.entity(entity).despawn();
    }
}

/// Spawnea un nodo y sus hijos recursivamente. Retorna la entidad raiz.
fn spawn_node_recursive(
    commands: &mut Commands,
    node: &InxNode,
    _parent_transform: Transform,
) -> Entity {
    let final_transform = Transform {
        translation: Vec3::new(
            node.transform.translation.x,
            -node.transform.translation.y,
            node.transform.translation.z,
        ),
        rotation: Quat::from_rotation_z(-node.transform.rotation.z),
        scale: Vec3::new(node.transform.scale.x, node.transform.scale.y, 1.0),
    };

    let base_opacity = node.material.as_ref().map(|m| m.opacity).unwrap_or(1.0);

    let entity = if let Some(mat) = &node.material {
        let mut ec = commands.spawn((
            InxUUID(node.uuid),
            InxZSort(node.zsort),
            node.node_type,
            mat.clone(),
            final_transform,
            if node.enabled {
                Visibility::Visible
            } else {
                Visibility::Hidden
            },
            Name::new(node.name.to_string()),
            InxBasePose {
                translation: final_transform.translation,
                rotation: final_transform.rotation,
                scale: final_transform.scale,
                opacity: base_opacity,
            },
        ));
        if let Some(mesh) = &mat.mesh {
            ec.insert(InxDeform {
                offsets: vec![[0.0, 0.0]; mesh.vertex_buffer.len()],
            });
        }
        ec.id()
    } else {
        commands
            .spawn((
                InxUUID(node.uuid),
                InxZSort(node.zsort),
                node.node_type,
                final_transform,
                if node.enabled {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                },
                Name::new(node.name.to_string()),
                InxBasePose {
                    translation: final_transform.translation,
                    rotation: final_transform.rotation,
                    scale: final_transform.scale,
                    opacity: 1.0,
                },
            ))
            .id()
    };

    // SimplePhysics
    if node.node_type == InxNodeType::SimplePhysics
        && let Some(phys) = &node.physics_data
    {
        if phys.param_uuid != u32::MAX {
            commands.entity(entity).insert((
                InxSimplePhysics {
                    param_uuid: phys.param_uuid,
                    model: phys.model,
                    map_mode: phys.map_mode,
                    gravity: phys.gravity,
                    length: phys.length,
                    frequency: phys.frequency,
                    angle_damping: phys.angle_damping,
                    length_damping: phys.length_damping,
                    output_scale: phys.output_scale,
                    local_only: phys.local_only,
                },
                InxPhysicsState::default(),
                GlobalTransform::default(),
            ));
        }
    }

    // Hijos — recursion directa sobre Vec<InxNode>
    for child_node in &node.children {
        let child_entity = spawn_node_recursive(commands, child_node, Transform::IDENTITY);
        commands.entity(entity).add_child(child_entity);
    }

    entity
}

/// Inicializa InxParamState con los defaults del puppet.
fn init_param_state(puppet: &InxPuppet, params: &Assets<InxParam>) -> InxParamState {
    let mut state = InxParamState::default();
    for handle in &puppet.params {
        if let Some(param) = params.get(handle) {
            state.values.insert(param.uuid, param.defaults);
        }
    }
    state
}
