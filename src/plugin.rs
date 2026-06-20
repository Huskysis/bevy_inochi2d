//! Bevy plugin wiring: registers assets, loaders and update systems.
//!
//! [`Inochi2dPlugin`] installs the render pipeline plus the spawn/prop sync
//! systems; [`InxAnimationPlugin`] adds the animation controller, param
//! evaluation and simple-physics systems in the correct order.

use bevy::prelude::*;

use crate::{
    InxAnimation, InxParam, InxPuppet,
    animation::{evaluate_params, update_animation_controller},
    auto_spawn::spawn_scene_system,
    sync_props,
    inr_loader::InrLoader,
    pipeline::InxRenderPipeline,
    simple_physics::{PhysicsEnabled, simple_physics_system},
};

pub struct Inochi2dPlugin;

impl Plugin for Inochi2dPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(InxRenderPipeline)
            .init_asset::<InxPuppet>()
            .init_asset::<InxParam>()
            .init_asset::<InxAnimation>()
            .preregister_asset_loader::<InrLoader>(&["inr"])
            .add_systems(Update, (spawn_scene_system, sync_props));
        #[cfg(feature = "inx")]
        app.preregister_asset_loader::<crate::loader::InxLoader>(&["inx", "inp"]);
    }

    fn finish(&self, app: &mut App) {
        app.register_asset_loader(InrLoader);
        #[cfg(feature = "inx")]
        app.register_asset_loader(crate::loader::InxLoader);
    }
}

pub struct InxAnimationPlugin;

impl Plugin for InxAnimationPlugin {
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
