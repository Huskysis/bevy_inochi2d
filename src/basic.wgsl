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
    offset: vec2<f32>,
    opacity: f32,
    mask_threshold: f32,
    emission_strength: f32,
    tint: vec3<f32>,
    screen_tint: vec3<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@group(2) @binding(0)
var tex_albedo: texture_2d<f32>;
@group(2) @binding(1)
var samp_albedo: sampler;

@group(3) @binding(0)
var tex_emissive: texture_2d<f32>;
@group(3) @binding(1)
var samp_emissive: sampler;

@group(4) @binding(0)
var tex_bumpmap: texture_2d<f32>;
@group(4) @binding(1)
var samp_bumpmap: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) deform: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

// MRT output - 3 render targets
struct FragmentOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) bump: vec4<f32>,
}

@vertex
fn vx_part(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) deform: vec2<f32>,
) -> VertexOutput {
    let pos = position - uniforms.offset + deform;
    let world_pos = uniforms.mvp * vec4<f32>(pos, 0.0, 1.0);
    let clip_pos = view.view_proj * world_pos;

    var out: VertexOutput;

    out.clip_position = clip_pos;
    out.uv = uv;
    
    return out;
}

@fragment
fn fg_part(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    // Textures are straight alpha; the sRGB view decodes to linear on sample.
    // Premultiply here so blending stays premultiplied in linear space.
    let albedo_tex = textureSample(tex_albedo, samp_albedo, in.uv);
    let albedo_sample = vec4<f32>(albedo_tex.rgb * albedo_tex.a, albedo_tex.a);

    let screen_blend = uniforms.screen_tint * albedo_sample.a;
    let screen_out = vec3<f32>(1.0) - ((vec3<f32>(1.0) - albedo_sample.rgb) * (vec3<f32>(1.0) - screen_blend));

    out.albedo = vec4<f32>(screen_out * uniforms.tint, albedo_sample.a) * uniforms.opacity;
    
    // Emissive
    let emissive_sample = textureSample(tex_emissive, samp_emissive, in.uv);
    out.emissive = vec4<f32>(emissive_sample.rgb * emissive_sample.a * uniforms.emission_strength, 1.0);
    
    // Bumpmap
    let bump_sample = textureSample(tex_bumpmap, samp_bumpmap, in.uv);
    out.bump = vec4<f32>(bump_sample.rgb, 1.0) * out.albedo.a;
    
    return out;
}
