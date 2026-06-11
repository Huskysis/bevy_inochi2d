## ✦ Bevy Inochi2d

Standalone Inochi2D renderer powered by Bevy's wgpu backend.

> **⚠️ Important:** This plugin is a **standalone rendering pipeline** - it is **NOT** designed to integrate with Bevy's rendering ecosystem. Bevy's render graph orchestration (sort-key ordered phases, batching, RenderPhase, RenderCommand) conflicts with Inochi2D's requirements: strict z-sort draw order, non-sRGB texture format, composite and mask stack in a sequential pass. This bypasses those systems entirely in a custom `ViewNode` that runs its own command list.

## ✦ Description

**bevy_inochi2d** loads .inr puppet files (and optionally .inx / .inp) and renders them through a fully custom wgpu pipeline inside Bevy's render graph. It leverages Bevy for windowing, asset system and ECS: but all draw calls, blend states and render targets are managed internally.

## ✦ Formats

| Format | Default | Notes |
| ------ | ------- | ----- |
| `.inr` | ✓ | Runtime format: flat pre-order node list, index cross-references, raw RGBA8 textures. Loads with **no image decoding and no UUID maps** — the self-contained reader only needs `serde_json` + `bytemuck`. |
| `.inx` / `.inp` | feature `inx` | Authoring formats. Pulls in `inochi2d-parser` and the `bevy_image` PNG/TGA decoders. |

Convert an authoring file to INR with the `inochi2d-inr` exporter:

```sh
cargo run -p inochi2d-inr --example inx2inr -- "Arch Chan.inx" "Arch Chan.inr"
```

INR files are larger on disk than INX (textures are stored as raw RGBA8
instead of PNG) in exchange for much faster, decode-free loading.

```toml
# INX/INP support is opt-in:
bevy_inochi2d = { version = "0.2", features = ["inx"] }
```

## ✦ Features

✦ **Asset loader**: Loads `.inr` files via Bevy's `AssetServer` (plus `.inx` / `.inp` with the `inx` feature): puppet tree, meshes, textures, parameters and animations.  
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
bevy_inochi2d = "0.2"
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

    let puppet: Handle<InxPuppet> = asset_server.load("my_puppet.inr");
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
| 0.2           | 0.17 |
| 0.1           | 0.17 |

**Note:** For Bevy 0.18, I'm currently exploring and experimenting with alternatives and strategies to integrate this plugin into the ecosystem.

## ✦ Dependencies

- `serde` / `serde_json`: INR JSON chunk parsing.
- `bytemuck`: Struct to bytes conversion (or GPU buffer).
- `bevy`: Windowing, asset system, ECS, easy access to wgpu.
- With feature `inx` only:
  - [`inochi2d-parser`](https://github.com/Huskysis/inochi2d-parser): IR parser for the INX/INP format.
  - `bevy_image`: PNG/TGA texture decoding.

## ✦ TODO

- [ ] Refactor and explore alternatives to pipeline.rs to make use of the adjacent bevy ecosystem.

## ✦ Example Asset

The example Puppet (Arch Chan.inx) was obtained from the [arch-chan](https://github.com/Speykious/arch-chan) repository under the CC0 1.0 Universal license.
