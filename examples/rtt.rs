//! Render-to-texture example.
//!
//! An offscreen camera (on `RenderLayers` layer 1) renders the puppet into an
//! `Image`. The main camera (layer 0) never sees the puppet directly - it only shows
//! the resulting texture, used both as a world `Sprite` and as a UI `ImageNode`
//! (e.g. a dialog portrait / HUD avatar).
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
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_inochi2d - rtt".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Inochi2dPlugin)
        .add_plugins(Inochi2dAnimationPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
) {
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

    // Offscreen camera: renders only layer 1 into the image, before the main camera
    // (order -1). No `Hdr` component, so the target stays Rgba8UnormSrgb.
    commands.spawn((
        Camera2d,
        Camera {
            order: -1,
            clear_color: ClearColorConfig::Custom(Color::NONE),
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
            default_pose: false,
        },
        RenderLayers::layer(1),
    ));

    // Main camera (default layer 0): sees only the texture consumers below.
    commands.spawn(Camera2d);

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
        Text::new("Node UI With Image")
    ));
}
