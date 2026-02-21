
struct View {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    viewport: vec4<f32>,
};

struct Uniforms {
    mvp: mat4x4<f32>,
    opacity: f32,
    tint: vec3<f32>,
    screen_tint: vec3<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@group(2) @binding(0) var tex_albedo: texture_2d<f32>;
@group(2) @binding(1) var samp_albedo: sampler;

@group(3) @binding(0) var tex_emissive: texture_2d<f32>;
@group(3) @binding(1) var samp_emissive: sampler;

@group(4) @binding(0) var tex_bumpmap: texture_2d<f32>;
@group(4) @binding(1) var samp_bumpmap: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vx_composite(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOutput {
    // Composite usa coordenadas de pantalla directamente
    // Requiere AABB de los hijos para determinar su tamaño
    // Tamaño sin hijos 1:1 sin view_proj sera fullscreen
    // con view_proj se reduce a ~0.001 aprox de tamaño(scale)
    // Estoy descartando el AABB, creo que es mejor fullscreen texture
    // let world_pos = uniforms.mvp * vec4<f32>(position, 0.0, 1.0);
    // let clip_pos = view.view_proj * world_pos;
    
    var out: VertexOutput;
    // out.clip_position = clip_pos;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = uv;
    
    return out;
}

struct FragmentOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) bump: vec4<f32>,
}

@fragment
fn fg_composite(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    let albedo_sample = textureSample(tex_albedo, samp_albedo, in.uv);
    let emissive_sample = textureSample(tex_emissive, samp_emissive, in.uv);
    let bump_sample = textureSample(tex_bumpmap, samp_bumpmap, in.uv);

    let screen_blend = uniforms.screen_tint * albedo_sample.a;
    let screen_out = vec3<f32>(1.0) - ((vec3<f32>(1.0) - albedo_sample.rgb) * (vec3<f32>(1.0) - screen_blend));

    out.albedo = vec4<f32>(screen_out * uniforms.tint, albedo_sample.a) * uniforms.opacity;

    out.emissive = emissive_sample * uniforms.opacity;

    out.bump = vec4<f32>(bump_sample.rgb, 1.0) * out.albedo.a;

    return out;
}
