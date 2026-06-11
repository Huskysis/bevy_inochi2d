use bevy::prelude::*;

use crate::{
    InxAnimation, InxParam, InxPuppet,
    animation::{evaluate_params, update_animation_controller},
    auto_spawn::spawn_scene_system,
    inr_loader::InrLoader,
    pipeline::InxRenderPipeline,
    simple_physics::simple_physics_system,
};

pub struct Inochi2dPlugin;

impl Plugin for Inochi2dPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(InxRenderPipeline)
            .init_asset::<InxPuppet>()
            .init_asset::<InxParam>()
            .init_asset::<InxAnimation>()
            .preregister_asset_loader::<InrLoader>(&["inr"])
            .add_systems(Update, spawn_scene_system);
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
        app.add_systems(
            Update,
            (
                simple_physics_system,
                update_animation_controller,
                evaluate_params,
            )
                .chain(),
        );
    }
}
