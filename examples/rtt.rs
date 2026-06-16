//! Render-to-texture example.
//!
//! An offscreen camera (on `RenderLayers` layer 1) renders the puppet into an
//! `Image`. The main camera (layer 0) never sees the puppet directly — it only
//! shows the resulting texture, used both as a world `Sprite` and as a UI
//! `ImageNode` (e.g. a dialog portrait / HUD avatar).
//!
//! Run with: `cargo run --example rtt`

use bevy::{
    asset::RenderAssetUsages,
    camera::{RenderTarget, visibility::RenderLayers},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

use bevy_inochi2d::prelude::*;

const RTT_SIZE: u32 = 1024;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Inochi2dPlugin)
        .add_plugins(InxAnimationPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, camera_controls)
        .run();
}

#[derive(Component)]
struct PrincipalCamera;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
) {
    // UI controls
    commands
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            ..Default::default()
        })
        .with_children(|child| {
            child.spawn(Text::new(
                "Press W/A/S/D/ or Arrow Up/Left/Down/Right to move camera",
            ));
            child.spawn(Text::new("Press + / - to zoom camera"));
            child.spawn(Text::new("Press Space to reset camera"));
        });

    // Offscreen render target. Must be sRGB + RENDER_ATTACHMENT | TEXTURE_BINDING.
    let size = Extent3d {
        width: RTT_SIZE,
        height: RTT_SIZE,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT;
    let image_handle = images.add(image);

    // Offscreen camera: renders only layer 1 into the image, before the main
    // camera (order -1). No `Hdr` component, so the target stays Rgba8UnormSrgb.
    commands.spawn((
        PrincipalCamera,
        Camera2d,
        Camera {
            order: -1,
            clear_color: ClearColorConfig::Custom(Color::srgb_u8(50, 50, 50)),
            ..Default::default()
        },
        RenderTarget::Image(image_handle.clone().into()),
        RenderLayers::layer(1),
        Transform::from_translation(Vec3::new(0.0, 790.0, 0.0)),
    ));

    // Puppet on layer 1: only the offscreen camera draws it.
    let puppet: Handle<InxPuppet> = asset_server.load("Arch Chan.inr");
    commands.spawn((
        InxScene {
            puppet,
            transform: Transform::from_scale(Vec3::splat(0.5)),
            animation: true,
        },
        RenderLayers::layer(1),
    ));

    // Main camera (default layer 0): sees only the texture consumers below.
    commands.spawn((
        Camera2d,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb_u8(90, 90, 90)),
            ..Default::default()
        },
    ));

    // The rendered texture as a world sprite...
    commands.spawn((
        Sprite {
            image: image_handle.clone(),
            custom_size: Some(Vec2::splat(512.0)),
            ..Default::default()
        },
        Transform::from_translation(Vec3::new(-150.0, 0.0, 0.0)),
    ));

    // ...and as a UI portrait in the corner.
    commands.spawn((
        ImageNode::new(image_handle),
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(16.0),
            bottom: Val::Px(16.0),
            width: Val::Px(256.0),
            height: Val::Px(256.0),
            ..Default::default()
        },
        Text::new("Node UI With Image"),
    ));
}

fn camera_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut camera: Query<(&mut Transform, &mut Projection), With<PrincipalCamera>>,
) {
    let Ok((mut transform, mut projections)) = camera.single_mut() else {
        return;
    };

    let speed = 500.0 * time.delta_secs();
    let zoom_speed = 1.0 * time.delta_secs();

    // Movement
    if keyboard.pressed(KeyCode::KeyA) | keyboard.pressed(KeyCode::ArrowLeft) {
        transform.translation.x -= speed;
    }
    if keyboard.pressed(KeyCode::KeyD) | keyboard.pressed(KeyCode::ArrowRight) {
        transform.translation.x += speed;
    }
    if keyboard.pressed(KeyCode::KeyW) | keyboard.pressed(KeyCode::ArrowUp) {
        transform.translation.y += speed;
    }
    if keyboard.pressed(KeyCode::KeyS) | keyboard.pressed(KeyCode::ArrowDown) {
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
