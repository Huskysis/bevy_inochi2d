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
    _opacity: f32,
    mask_threshold: f32,
    _emission_strength: f32,
    _mult_color: vec3<f32>,
    _screen_color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> view: View;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@group(2) @binding(0)
var tex_albedo: texture_2d<f32>;
@group(2) @binding(1)
var samp_albedo: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vx_mask(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOutput {
    let pos = position - uniforms.offset;
    let world_pos = uniforms.mvp * vec4<f32>(pos, 0.0, 1.0);
    let clip_pos = view.view_proj * world_pos;

    var out: VertexOutput;
    out.clip_position = clip_pos;
    out.uv = uv;
    return out;
}

@fragment
fn fg_mask(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(tex_albedo, samp_albedo, in.uv);
    if (tex_color.a < uniforms.mask_threshold) {
        // descarta todo lo que este fuera
        // de los bordes del que apunta
        discard;
    }

    return vec4(0.0);
}
