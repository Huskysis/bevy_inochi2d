//! Bevy `Plugin`s that register this crate's systems.

use bevy::prelude::*;

use crate::{
    InxAnimation, InxParam, InxPuppet,
    animation::{apply_default_pose, evaluate_params, update_animation_controller},
    auto_spawn::spawn_scene_system,
    composite::{CompositeRtPool, upgrade_composite_mode_for_dst_reading_children},
    sync_props,
    inr_loader::InrLoader,
    mesh2d::InxMesh2dPlugin,
    simple_physics::{PhysicsEnabled, simple_physics_system},
};

/// Assets, loaders and scene spawning - everything except a renderer. Use through
/// [`Inochi2dPlugin`] (Mesh2d/Material2d renderer), or add
/// [`crate::mesh2d::InxMesh2dPlugin`] yourself for finer control.
pub struct Inochi2dCorePlugin;

impl Plugin for Inochi2dCorePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<InxPuppet>()
            .init_asset::<InxParam>()
            .init_asset::<InxAnimation>()
            .init_resource::<CompositeRtPool>()
            .preregister_asset_loader::<InrLoader>(&["inr"])
            .add_systems(
                Update,
                (
                    spawn_scene_system,
                    sync_props,
                    upgrade_composite_mode_for_dst_reading_children.after(spawn_scene_system),
                    apply_default_pose,
                ),
            );
        #[cfg(feature = "inx")]
        app.preregister_asset_loader::<crate::loader::InxLoader>(&["inx", "inp"]);
    }

    fn finish(&self, app: &mut App) {
        app.register_asset_loader(InrLoader);
        #[cfg(feature = "inx")]
        app.register_asset_loader(crate::loader::InxLoader);
    }
}

/// Main entry point: [`Inochi2dCorePlugin`] + [`InxMesh2dPlugin`]. Add
/// [`Inochi2dAnimationPlugin`] alongside it for animation/physics playback.
pub struct Inochi2dPlugin;

impl Plugin for Inochi2dPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(Inochi2dCorePlugin)
            .add_plugins(InxMesh2dPlugin);
    }
}

/// Animation/physics playback loop (`update_animation_controller` -> `simple_physics_system` -> `evaluate_params`, in that order every frame).
/// Optional: a puppet spawned with `InxScene::default_pose = true` gets its rest
/// pose without this plugin at all.
pub struct Inochi2dAnimationPlugin;

impl Plugin for Inochi2dAnimationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PhysicsEnabled>().add_systems(
            Update,
            (
                update_animation_controller,
                simple_physics_system,
                evaluate_params,
            )
                .chain(),
        );
    }
}
