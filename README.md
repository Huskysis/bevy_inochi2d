## ✦ Bevy Inochi2d

Standalone Inochi2D renderer powered by Bevy's wgpu backend.

> **⚠️ Important:** This plugin is a **standalone rendering pipeline** - it is **NOT** designed to integrate with Bevy's rendering ecosystem. Bevy's render graph orchestration (sort-key ordered phases, batching, RenderPhase, RenderCommand) conflicts with Inochi2D's requirements: strict z-sort draw order, non-sRGB texture format, composite and mask stack in a sequential pass. This bypasses those systems entirely in a custom `ViewNode` that runs its own command list.

## ✦ Description

**bevy_inochi2d** loads .inx / .inp puppet files and renders them through a fully custom wgpu pipeline inside Bevy's render graph. It leverages Bevy for windowing, asset system and ECS: but all draw calls, blend states and render targets are managed internally.

## ✦ Features

✦ **Asset loader**: Loads `.inx` / `.inp` files via Bevy's `AssetServer`, parsing the puppet tree, meshes, textures (PNG/TGA), parameters and animations.  
✦ **Custom rendering pipeline**: A single `ViewNode` with its own vertex/index buffers, MRT (albedo + emissive + bumpmap) and a command list (`DrawPart`, `BeginComposite`/`EndComposite`, `PushMask`/`PopMask`).  
✦ **Mask system**: Stencil-based masks with Mask and Dodge modes.  
✦ **Composite nodes**: Offscreen render targets for grouped composition with opacity and tint.  
✦ **Parameter system**: 2D grid interpolation (linear/cubic/stepped) for transform bindings and mesh deformations.  
✦ **Animation controller**: Multi-layer with transition, looping and per-layer blend (additive/override).  
✦ **Simple physics**: Pendulum and spring-pendulum simulation feeding params (hair, accessories, etc).  
✦ **Scene spawn**: `InxScene` component to spawn a puppet automatically.

## ✦ Usage example

```toml
[dependencies]
bevy_inochi2d = "0.1"
```

```rust
use bevy::prelude::*;
use bevy_inochi2d::{InxScene, animation::InxAnimationPlugin, prelude::*};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(Inochi2dPlugin)
        .add_plugins(InxAnimationPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2d);

    let puppet: Handle<InxPuppet> = asset_server.load("my_puppet.inx");
    commands.spawn(InxScene {
        puppet,
        transform: Transform::from_scale(Vec3::splat(0.5)),
        animation: true,
    });
}
```

## Why didn't I use Bevy's standard pipeline?

Inochi2D puppets require a **strict draw order** defined by the puppet tree (that is, the nodes that compose it), it can be achieved but the render does not reach the order, several `RenderPipeline` are required for each node with its own `RenderPass` (in bevy it would be `TrackedRenderPass`), for me it is too tangled, I opted for a monolithic system, for now. Adding to it, Bevy's `RenderPhase` system is designed for sort-key batching and parallelism, which breaks the sequential contract needed for heterogeneous structures. Integrating with `bevy_sprite` or `bevy_render`'s phase system would mean fighting the ecosystem at every step, so I decided to use its own `ViewNode` as a direct command list.

**What this implies in practice:**

- Inochi2D puppets render correctly with full spec compliance.
- Bevy's standard sprites/meshes **do not** interleave or depth-sort with puppet parts.
- The puppet renders as a single layer in Bevy's render graph (after `Node2d::MainPass`).

## Compatibility

| bevy_inochi2d | Bevy |
| ------------- | ---- |
| 0.1           | 0.17 |

**Note:** For Bevy 0.18, I'm currently exploring and experimenting with alternatives and strategies to integrate this plugin into the ecosystem.

## ✦ Dependencies

- [`inochi2d-parser`](https://github.com/Huskysis/inochi2d-parser): IR parser for the INX/INP format.
- `bytemuck`: Struct to bytes conversion (or GPU buffer).
- `bevy`: Windowing, asset system, ECS, easy access to wgpu.
- `bevy_image`: PNG/TGA texture decoding.

## ✦ TODO

- [ ] Refactor and explore alternatives to pipeline.rs to make use of the adjacent bevy ecosystem.

## ✦ Example Asset

The example Puppet (Arch Chan.inx) was obtained from the [arch-chan](https://github.com/Speykious/arch-chan) repository under the CC0 1.0 Universal license.
