//! Debug/inspection bench for the Mesh2d renderer: interleaving of a plain Bevy
//! sprite with puppet parts by Z, per-composite render-target dumps (F10) and a
//! physics on/off toggle (F9) for pose-clean captures.
//!
//! Run with: `cargo run --example mesh2d`

use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};

use bevy_inochi2d::composite::InxCompositeBbox;
use bevy_inochi2d::simple_physics::PhysicsEnabled;
use bevy_inochi2d::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(bevy::log::LogPlugin {
            level: bevy::log::Level::WARN,
            ..default()
        }).set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_inochi2d - mesh2d".into(),
                ..default()
            }),
            ..default()
        }))

        .add_plugins(Inochi2dPlugin)
        .add_plugins(Inochi2dAnimationPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (move_marker, camera_controls, dump_composite_rts, toggle_physics))
        .run();
}

#[derive(Component)]
struct PrincipalCamera;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((PrincipalCamera, Camera2d, Transform::from_translation(Vec3::new(0.0, 790.0, 0.0))));
    let puppet: Handle<InxPuppet> = asset_server.load("Arch Chan.inr");
    commands.spawn(InxScene {
        puppet,
        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0))
            .with_scale(Vec3::splat(0.5)),
        animation: true,
        default_pose: true
    });

    // Plain Bevy sprite. Its Z decides where it lands between puppet parts (parts use z = -zsort).
    // Move it with Up/Down.
    commands.spawn((
        Sprite::from_color(Color::srgb(0.9, 0.2, 0.3), Vec2::new(600.0, 70.0)),
        Transform::from_translation(Vec3::new(0.0, 850.0, 0.0)),
        ZMarker,
    ));

    println!("\nArrow Up/Down: move the red bar through the puppet's Z (interleaving test)");
    println!("F10: dump composite RTs to /tmp/rt_dump_<name>.png (debug readback)");
    println!("F9: toggle SimplePhysics on/off (pose-clean captures)\n");
}

/// F9: toggle global PhysicsEnabled - freezes hair/accessory sway for pose-clean
/// captures and visual comparisons.
fn toggle_physics(keyboard: Res<ButtonInput<KeyCode>>, mut enabled: ResMut<PhysicsEnabled>) {
    if !keyboard.just_pressed(KeyCode::F9) {
        return;
    }
    enabled.0 = !enabled.0;
    println!("PhysicsEnabled = {}", enabled.0);
}

/// F10: snapshot every live composite RT to /tmp as PNG (renderdoc-lite). Spawns a
/// one-shot `Readback` per RT; the observer strips row padding and encodes RGBA8.
/// The readback entity despawns itself after firing.
fn dump_composite_rts(
    keyboard: Res<ButtonInput<KeyCode>>,
    composites: Query<(&Name, &InxCompositeBbox)>,
    mut commands: Commands,
) {
    if !keyboard.just_pressed(KeyCode::F10) {
        return;
    }
    let mut count = 0;
    for (name, bbox) in composites.iter() {
        let Some(rt) = bbox.rt.clone() else { continue };
        let side = bbox.rt_side;
        let file = format!(
            "/tmp/rt_dump_{}{}.png",
            name.as_str().replace([' ', ':', '/'], "_"),
            count
        );
        commands
            .spawn(Readback::texture(rt))
            .observe(
                move |trigger: On<ReadbackComplete>, mut commands: Commands| {
                    let data = &trigger.event().data;
                    let unpadded = (side * 4) as usize;
                    let padded = data.len() / side as usize;
                    let mut pixels = Vec::with_capacity(unpadded * side as usize);
                    for row in 0..side as usize {
                        let start = row * padded;
                        pixels.extend_from_slice(&data[start..start + unpadded]);
                    }
                    match image::RgbaImage::from_raw(side, side, pixels) {
                        Some(img) => match img.save(&file) {
                            Ok(()) => println!("[rt-dump] wrote {file} ({side}x{side})"),
                            Err(e) => eprintln!("[rt-dump] save failed {file}: {e}"),
                        },
                        None => eprintln!("[rt-dump] bad buffer size for {file}"),
                    }
                    commands.entity(trigger.event().entity).despawn();
                },
            );
        count += 1;
    }
    println!("[rt-dump] requested {count} RT snapshot(s)");
}

#[derive(Component)]
struct ZMarker;

fn move_marker(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut marker: Query<&mut Transform, With<ZMarker>>,
) {
    let Ok(mut transform) = marker.single_mut() else {
        return;
    };
    if keyboard.pressed(KeyCode::ArrowUp) {
        transform.translation.z += 0.1;
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        transform.translation.z -= 0.1;
    }
}


fn camera_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut camera: Query<(&mut Transform, &mut Projection, &Camera2d), With<PrincipalCamera>>,
) {
    let Ok((mut transform, mut projections, _)) = camera.single_mut() else {
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
