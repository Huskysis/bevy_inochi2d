## ✦ Bevy Inochi2d

Standalone Inochi2D renderer powered by Bevy's wgpu backend.

> **⚠️ Important:** This plugin is a **standalone rendering pipeline** - it is **NOT** designed to integrate with Bevy's rendering ecosystem. Bevy's render graph orchestration (sort-key ordered phases, batching, RenderPhase, RenderCommand) conflicts with Inochi2D's requirements: strict z-sort draw order, composite and mask stack in a sequential pass. This bypasses those systems entirely in a custom `ViewNode` that runs its own command list.

## ✦ Description

**bevy_inochi2d** loads .inr puppet files (and optionally .inx / .inp) and renders them through a fully custom wgpu pipeline inside Bevy's render graph. It leverages Bevy for windowing, asset system and ECS: but all draw calls, blend states and render targets are managed internally.

## ✦ Formats

| Format | Default | Notes |
| ------ | ------- | ----- |
| `.inr` | ✓ | Runtime format: flat pre-order node list, index cross-references, raw RGBA8 textures. Loads with **no image decoding and no UUID maps** via `inochi2d-parser`'s `inr` feature (`serde_json` + `bytemuck` only). |
| `.inx` / `.inp` | feature `inx` | Authoring formats. Pulls in the full IR parser and the `bevy_image` PNG/TGA decoders. |

Convert an authoring file to INR with the parser's exporter (run inside the
[`inochi2d-parser`](https://github.com/Huskysis/inochi2d-parser) repo):

```sh
cargo run --features inr-export --example inx2inr -- "Arch Chan.inx" "Arch Chan.inr"
```

INR files are larger on disk than INX (textures are stored as raw RGBA8
instead of PNG) in exchange for much faster, decode-free loading.

```toml
# INX/INP support is opt-in:
bevy_inochi2d = { version = "0.3", features = ["inx"] }
```

## ✦ Features

✦ **Asset loader**: Loads `.inr` files via Bevy's `AssetServer` (plus `.inx` / `.inp` with the `inx` feature): puppet tree, meshes, textures, parameters and animations.  
✦ **Custom rendering pipeline**: A single `ViewNode` with its own vertex/index buffers drawing straight to the view target with a command list (`DrawPart`, `BeginComposite`/`EndComposite`, `PushMask`/`PopMask`).  
✦ **Mask system**: Stencil-based masks with Mask and Dodge modes.  
✦ **Composite nodes**: Offscreen render targets for grouped composition with opacity and tint.  
✦ **Parameter system**: 2D grid interpolation (linear/cubic/stepped) for transform bindings and mesh deformations.  
✦ **Animation controller**: Multi-layer with transition, looping and per-layer blend (additive/override).  
✦ **Simple physics**: Pendulum and spring-pendulum simulation feeding params (hair, accessories, etc).  
✦ **Scene spawn**: `InxScene` component to spawn a puppet automatically.  
✦ **Props (pseudo-sprites)**: `InxProp` attaches external textured quads to puppet nodes, rendered *inside* the pipeline with correct z-ordering between parts (e.g. an item in the hand).  
✦ **Render to texture**: Works with Bevy's standard `RenderTarget::Image` and `RenderLayers` — render a puppet offscreen and use the result anywhere (UI, sprites, 3D).

## ✦ Usage example

```toml
[dependencies]
bevy_inochi2d = "0.3"
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
| 0.3           | 0.18 |
| 0.2           | 0.17 |
| 0.1           | 0.17 |

**Note:** Since Bevy 0.18 the `2d` feature collection officially supports bringing your own 2D renderer, which is exactly what this plugin does.

## ✦ Dependencies

- [`inochi2d-parser`](https://github.com/Huskysis/inochi2d-parser) (feature `inr`): INR container reading and typed document.
- `bytemuck`: Struct to bytes conversion (or GPU buffer).
- `bevy`: Windowing, asset system, ECS, easy access to wgpu.
- With feature `inx` only:
  - `inochi2d-parser` IR types for the INX/INP authoring format.
  - `bevy_image`: PNG/TGA texture decoding.

## ✦ Props: attaching external content

A puppet is not a closed quad: parent an `InxProp` to any puppet node and it renders
inside the custom pipeline, z-sorted between puppet parts. Props are regular ECS
entities (transforms, collisions, queries all work); only the drawing goes through the
plugin's ViewNode — a plain Bevy `Sprite` still renders above/below the whole puppet.

```rust
// find a node entity (e.g. by Name), then:
commands.entity(hand).with_child((
    InxProp {
        texture: asset_server.load("sword.png"),
        size: Vec2::new(64.0, 200.0),
        ..Default::default()
    },
    InxZSort(-0.05), // lower zsort = in front of the hand
));
```

See `examples/prop.rs` for a runnable demo (attach/remove at runtime).

## ✦ Render to texture

The pipeline renders per view, so Bevy's standard render-to-texture just works:
point a camera at an `Image` asset with `RenderTarget::Image` and put the puppet on
a dedicated `RenderLayers` layer so only that camera draws it. `RenderLayers` can be
set directly on the `InxScene` command (it is propagated to the puppet root).

```rust
// Offscreen camera draws layer 1 into `image_handle`.
commands.spawn((
    Camera2d,
    Camera { order: -1, ..Default::default() },
    RenderTarget::Image(image_handle.clone().into()),
    RenderLayers::layer(1),
));

// Puppet visible only to that camera.
commands.spawn((
    InxScene { puppet, transform, animation: true },
    RenderLayers::layer(1),
));
```

The resulting texture is a regular `Handle<Image>`: use it in UI (`ImageNode`),
as a `Sprite`, on a 3D mesh, etc. This is the cheap interop bridge while parts
don't interleave with Bevy sprites — typical uses:

- Dialog portraits / HUD avatars (`ImageNode`).
- Character select screens, in-game screens and mirrors.
- 3D billboards (puppet on a quad in a 3D world).
- Per-puppet post-processing (outline, hit-flash) on the texture.
- Rendering at a resolution decoupled from the window; thumbnails.

Notes: the target image must be `Rgba8UnormSrgb` with
`RENDER_ATTACHMENT | TEXTURE_BINDING` usage, and the offscreen camera must stay
LDR (no `Hdr` component). See `examples/rtt.rs` for a runnable demo.

## ✦ TODO

- [ ] **Long-term direction**: migrate rendering to `Mesh2d`/`Material2d` (the
      bevy_spine model) so Bevy sprites interleave natively with puppet parts. This
      requires reworking masks as CPU polygon clipping, emulating composites and the
      extended blend modes, and evolving the INR format to match the Mesh2d-friendly
      layout. Definitive ecosystem integration; planned after `InxProp` proves the
      interaction model.

## ✦ Example Asset

The example Puppet (Arch Chan.inx) was obtained from the [arch-chan](https://github.com/Speykious/arch-chan) repository under the CC0 1.0 Universal license.
