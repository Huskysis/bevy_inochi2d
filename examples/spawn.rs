use bevy::prelude::*;

use bevy_inochi2d::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Inochi2dPlugin)
        .add_plugins(InxAnimationPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, playback_animation)
        .add_systems(Update, camera_controls)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Camara
    commands.spawn((
        Camera2d::default(),
        Camera {
            ..Default::default()
        },
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
    ));

    // Cargar y spawnear puppet
    let puppet: Handle<InxPuppet> = asset_server.load("Arch Chan.inr");
    commands.spawn(InxScene {
        puppet,
        transform: Transform::from_scale(Vec3::splat(0.5)),
        animation: true,
    });

    println!("\n\nPress W/A/S/D to move camera");
    println!("Press + / - to zoom camera");
    println!("Press Space to reset camera");
    println!("Press Q to stop all animations");
    println!("Press E to play animation\n\n");
}

fn playback_animation(
    mut query: Query<(&InxPuppetRoot, &mut InxAnimationController)>,
    asset_inx: Res<Assets<InxPuppet>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    for (puppet_handle, mut controller) in query.iter_mut() {
        if keyboard.just_pressed(KeyCode::KeyQ) {
            controller.stop_all();
        }

        if keyboard.just_pressed(KeyCode::KeyE) {
            if let Some(puppet) = asset_inx.get(&puppet_handle.source)
                && let Some(animation) = puppet.named_animations.get("headpos")
            {
                controller.play_looped(animation.clone(), 0.3);
            }
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
