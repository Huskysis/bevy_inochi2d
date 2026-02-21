## ✦ Bevy Inochi2d

Renderizador independiente de Inochi2D impulsado por el backend wgpu de Bevy.

> **⚠️ Importante:** Este plugin es un **pipeline de renderizado propio** - **NO** está diseñado para integrarse con el ecosistema de renderizado de Bevy. El sistema de orquestación del render graph de Bevy (fases ordenadas por sort-key, batching, RenderPhase, RenderCommand) entra en conflicto con los requisitos de Inochi2D: orden de dibujado estricto por z-sort, formato de textura no sRGB, stack de composites y masks en un pase secuencial. Esto bypasea esos sistemas por completo en `ViewNode` custom que ejecuta su propia lista de comandos.

## ✦ Descripción

**bevy_inochi2d** carga archivos puppet .inx / .inp y los renderiza mediante un pipeline wgpu completamente custom dentro del render graph de Bevy. Aprovecha Bevy para ventana, sistema de assets y ECS: pero todos los draw calls, blend states y render targets se gestionan internamente.

## ✦ Características

✦ **Asset loader**: Carga archivos `.inx` / `.inp` vía `AssetServer` de Bevy, parseando el árbol del puppet, meshes, texturas (PNG/TGA), parámetros y animaciones.  
✦ **Pipeline de renderizado custom**: Un solo `ViewNode` con sus propios vertex/index buffers, MRT (albedo + emissive + bumpmap) y una lista de comandos (`DrawPart`, `BeginComposite`/`EndComposite`, `PushMask`/`PopMask`).  
✦ **Sistema de máscaras**: Máscaras basadas en stencil con modos Mask y Dodge.  
✦ **Nodos composite**: Render targets offscreen para composición agrupada con opacidad y tint.  
✦ **Sistema de parámetros**: Interpolación en grilla 2D (linear/cubic/stepped) para bindings de transform y deformaciones de mesh.  
✦ **Controlador de animación**: Multi-capa con transición, looping y blend por capa (additive/override).  
✦ **Física simple**: Simulación de péndulo y spring-pendulum que alimentan params (pelo, accesorios, etc).  
✦ **Spawn de escenas**: Componente `InxScene` para spawnear un puppet automaticamente.

## ✦ Ejemplo de uso

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

    let puppet: Handle<InxPuppet> = asset_server.load("mi_puppet.inx");
    commands.spawn(InxScene {
        puppet,
        transform: Transform::from_scale(Vec3::splat(0.5)),
        animation: true,
    });
}
```

## ¿Por qué no utilizé el pipeline estándar de Bevy?

Los puppets de Inochi2D requieren un **orden de dibujado estricto** definido por el árbol del puppet(es decir, los nodos que lo componen), se puede lograr pero en el render no alcanza el orden, se requiere varios `RenderPipeline` para cada nodo con su propio `RenderPass`(en bevy seria `TrackedRenderPass`), para mi es muy enredado, opte por un sistema monolítico, por ahora. Añadiendo, el sistema `RenderPhase` de Bevy está diseñado para batching por sort-key y paralelismo, lo que rompe el contrato secuencial para utilizar estructuras heterogéneas. Integrarse con `bevy_sprite` o el sistema de fases de `bevy_render` implicaría pelear contra el ecosistema en cada paso, así que me decidí usar su propio `ViewNode` como una lista de comandos directos.

**Qué implica en la práctica:**

- Los puppets Inochi2D se renderizan correctamente con cumplimiento completo de la spec.
- Los sprites/meshes estándar de Bevy **no** se intercalan ni hacen depth-sort con las partes del puppet.
- El puppet se renderiza como una capa única en el render graph de Bevy (después de `Node2d::MainPass`).

## Compatibilidad

| bevy_inochi2d | Bevy |
| ------------- | ---- |
| 0.1           | 0.17 |

## ✦ Dependencias

- [`inochi2d-parser`](https://github.com/Huskysis/inochi2d-parser): Parser IR del formato INX/INP.
- `bytemuck`: Conversión de struct a bytes(o buffer GPU).
- `bevy`: Ventana, sistema de assets, ECS, acceso fácil a wgpu.
- `bevy_image`: Decodificación de texturas PNG/TGA.

## ✦ Por hacer (TODO)

- [ ] Refactorizar y explorar alternativas a pipeline.rs para poder utilizar el ecosistema de bevy adyacente.

## ✦ Activo de Ejemplo

El Puppet de ejemplo (Arch Chan.inx) lo obtuve del repositorio [arch-chan](https://github.com/Speykious/arch-chan) bajo la licencia CC0 1.0 Universal.
