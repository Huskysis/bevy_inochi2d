//! Attach external content (a "prop") to a specific puppet NODE.
//!
//! Press P: attach a red quad to a node (in front of it, correct zsort). Press O:
//! remove the prop. The prop is a regular ECS entity; it follows the node's
//! transform (including param-driven motion, so it moves with hair/hand/etc) and
//! renders as a normal `Mesh2d`, z-sorted against puppet parts by Bevy.
//!
//! `InxProp` is NOT the only way to add external content - parts already interleave
//! natively with plain Bevy sprites/meshes by Z (see `examples/mesh2d.rs`'s `ZMarker`, a bare `Sprite` with no InxProp at all).
//! Use `InxProp` specifically when the content must ATTACH to and follow a puppet
//! node's own animated/physics-driven transform; for an independent object just
//! placed at some Z, a plain Bevy entity is enough.

use bevy::prelude::*;
use bevy_inochi2d::{InxProp, InxZSort, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Inochi2dPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (list_nodes_once, toggle_prop, change_zsort_prop))
        .add_systems(Update, camera_controls)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((Camera2d, Transform::from_translation(Vec3::new(0.0, 790.0, 0.0))));

    let puppet: Handle<InxPuppet> = asset_server.load("Arch Chan.inr");
    commands.spawn(InxScene {
        puppet,
        transform: Transform::from_scale(Vec3::splat(0.5)),
        animation: false,
        default_pose: true,
    });
}

fn list_nodes_once(
    nodes: Query<(Entity, &Name), Added<InxUUID>>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    let mut any = false;
    for (entity, name) in nodes.iter() {
        println!("node {entity:?}: {name}");
        any = true;
    }
    if any {
        *done = true;
        println!("\nPress P/O to attach/remove prop");
        println!("\nPress Arrow Up/Down increase/decrease Z sort\n (+Z = more backward, -Z = more forward)");
    }
}

#[derive(Component)]
struct DemoProp;

fn toggle_prop(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    nodes: Query<(Entity, &Name), With<InxUUID>>,
    props: Query<Entity, With<DemoProp>>,
    mut images: ResMut<Assets<Image>>,
    mut auto_done: Local<bool>,
) {
    let auto = !*auto_done && !nodes.is_empty();
    if (keyboard.just_pressed(KeyCode::KeyP) || auto) && props.is_empty() {
        *auto_done = true;
        // Pick a target node by name (fall back to the first named node).
        let target = nodes
            .iter()
            .find(|(_, n)| n.as_str().contains("Head"))
            .or_else(|| nodes.iter().next())
            .map(|(e, n)| (e, n.to_string()));

        let Some((target, name)) = target else {
            println!("puppet not spawned yet");
            return;
        };

        let red = images.add(Image::new_fill(
            bevy::render::render_resource::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            bevy::render::render_resource::TextureDimension::D2,
            &[255, 0, 0, 255],
            bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            bevy::asset::RenderAssetUsages::RENDER_WORLD,
        ));

        println!("attaching prop to {name}");
        commands.entity(target).with_child((
            DemoProp,
            InxProp {
                texture: red,
                size: Vec2::new(1000.0, 150.0),
                ..Default::default()
            },
            // lower zsort = drawn later = in front of the target node
            InxZSort(-40.0),
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));
    }

    if keyboard.just_pressed(KeyCode::KeyO) {
        for prop in props.iter() {
            println!("removing prop");
            commands.entity(prop).despawn();
        }
    }
}

fn change_zsort_prop(
    mut query: Query<&mut InxZSort, With<DemoProp>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    for mut z in query.iter_mut() {
        if keyboard.pressed(KeyCode::ArrowDown) {
            z.0 += 0.5;
            println!("Prop Z: {}", z.0);
        }
        if keyboard.pressed(KeyCode::ArrowUp) {
            z.0 -= 0.5;
            println!("Prop Z: {}", z.0);
        }
    }
}


fn camera_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut camera: Query<(&mut Transform, &mut Projection), With<Camera2d>>,
) {
    let Ok((mut transform, mut projections)) = camera.single_mut() else {
        return;
    };

    let speed = 500.0 * time.delta_secs();
    let zoom_speed = 1.0 * time.delta_secs();

    // Movement
    if keyboard.pressed(KeyCode::KeyA) {
        transform.translation.x -= speed;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        transform.translation.x += speed;
    }
    if keyboard.pressed(KeyCode::KeyW) {
        transform.translation.y += speed;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        transform.translation.y -= speed;
    }

    if let Projection::Orthographic(projection) = &mut *projections {
        // Zoom
        if keyboard.pressed(KeyCode::Equal) || keyboard.pressed(KeyCode::NumpadAdd) {
            projection.scale = (projection.scale - zoom_speed).max(0.1);
        }
        if keyboard.pressed(KeyCode::Minus) || keyboard.pressed(KeyCode::NumpadSubtract) {
            projection.scale = (projection.scale + zoom_speed).min(10.0);
        }

        // Reset
        if keyboard.just_pressed(KeyCode::Space) {
            transform.translation = Vec3::new(0.0, 0.0, 1000.0);
            projection.scale = 1.0;
        }
    }
}
