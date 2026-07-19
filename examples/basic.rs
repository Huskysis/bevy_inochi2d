//! Minimal puppet spawn: load, animate, camera controls.
//!
//! Demonstrates the animation controller (crossfade, loop, pause/resume, stop-with-reset)
//! and per-puppet physics on/off. Start here.

use bevy::prelude::*;

use bevy_inochi2d::{
    animation::{evaluate_params, update_animation_controller},
    prelude::*,
    simple_physics::InxPuppetPhysicsEnabled,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Inochi2dPlugin)
        .add_plugins(Inochi2dAnimationPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, playback_animation)
        .init_resource::<EyesFollowMouse>()
    // Must run between the controller (which rewrites every param default each frame)
    // and param evaluation, or the inserted values get stomped.
    .add_systems(
        Update,
        eyes_follow_mouse
            .after(update_animation_controller)
            .before(evaluate_params),
    )
    .add_systems(Update, toggle_local_physics)
    .add_systems(Update, camera_controls)
    .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Camera
    commands.spawn((
        Camera2d,
        Transform::from_translation(Vec3::new(0.0, 790.0, 0.0)),
    ));

    let puppet: Handle<InxPuppet> = asset_server.load("Arch Chan.inr");
    commands.spawn(InxScene {
        puppet,
        transform: Transform::from_scale(Vec3::splat(0.5)),
        animation: true,
        default_pose: true,
    });

    println!("\n\nPress W/A/S/D to move camera");
    println!("Press + / - to zoom camera");
    println!("Press Space to reset camera");
    println!("Press Q to stop all animations (instant reset)");
    println!("Press R to stop with fade-out (eases back to default pose)");
    println!("Press E to play animation ('headpos', looped)");
    println!("Press P to toggle animation pause (freezes frame, no reset)");
    println!("Press L to toggle local physics override");
    println!("Press M to toggle eyes-follow-mouse");
    println!("\n");
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

        // Smooth stop: fade every playing layer out over 0.5s. As the weight drops,
        // blended params slide back to their defaults, so the puppet eases into its
        // rest pose instead of snapping like Q does. 
        // stop_actions() skips layer 0, where play_looped() put 'headpos' on an idle-less controller,
        //  so the fade is set directly.
        if keyboard.just_pressed(KeyCode::KeyR) {
            for layer in controller.layers.iter_mut() {
                if layer.playing && layer.weight > 0.0 {
                    layer.fade = FadeState::FadingOut {
                        duration: 0.5,
                        elapsed: 0.0,
                        start_weight: layer.weight,
                    };
                }
            }
        }

        if keyboard.just_pressed(KeyCode::KeyP) {
            if controller.paused {
                controller.resume();
            } else {
                controller.pause();
            }
        }

        let Some(puppet) = asset_inx.get(&puppet_handle.source) else {
            continue;
        };

        if keyboard.just_pressed(KeyCode::KeyE)
            && let Some(animation) = puppet.named_animations.get("headpos")
        {
            controller.play_looped(animation.clone(), 0.3);
        }
    }
}


// Eye params, verified against Arch Chan.inr: vec2, range [-1, 1].
const EYE_LEFT_XY: u32 = 4212764581;
const EYE_RIGHT_XY: u32 = 384253162;

/// Toggle (M key): when on, eyes track the cursor.
#[derive(Resource, Default)]
struct EyesFollowMouse(bool);

fn eyes_follow_mouse(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut enabled: ResMut<EyesFollowMouse>,
    windows: Query<&Window>,
    mut puppets: Query<&mut InxParamState, With<InxPuppetRoot>>,
) {
    if keyboard.just_pressed(KeyCode::KeyM) {
        enabled.0 = !enabled.0;
        println!("[eyes follow mouse] {}", if enabled.0 { "on" } else { "off" });
    }
    if !enabled.0 {
        return; // params fall back to defaults -> eyes recenter on their own
    }

    let Ok(window) = windows.single() else { return };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let size = Vec2::new(window.width(), window.height());
    // Normalize to [-1, 1]; cursor Y grows downward, param Y up.
    let n = ((cursor / size) * 2.0 - Vec2::ONE) * Vec2::new(1.0, -1.0);
    for mut state in puppets.iter_mut() {
        state.values.insert(EYE_LEFT_XY, [n.x, n.y]);
        state.values.insert(EYE_RIGHT_XY, [n.x, n.y]);
    }
}

// Demonstrates InxPuppetPhysicsEnabled (per-puppet local override of the global PhysicsEnabled resource)-
// L toggles it on the puppet root.
fn toggle_local_physics(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    roots: Query<(Entity, Option<&InxPuppetPhysicsEnabled>), With<InxPuppetRoot>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyL) {
        return;
    }
    for (entity, local) in roots.iter() {
        let next = !local.map(|l| l.0).unwrap_or(true);
        println!("[local physics] Arch Chan -> InxPuppetPhysicsEnabled({next})");
        commands
            .entity(entity)
            .insert(InxPuppetPhysicsEnabled(next));
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
