#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct PartMaterial {
    tint: vec3<f32>,
    opacity: f32,
    screen_tint: vec3<f32>,
    composite: u32,
    emissive_strength: f32,
}

@group(2) @binding(0) var<uniform> material: PartMaterial;
@group(2) @binding(1) var tex_albedo: texture_2d<f32>;
@group(2) @binding(2) var samp_albedo: sampler;
@group(2) @binding(3) var tex_emissive: texture_2d<f32>;
@group(2) @binding(4) var samp_emissive: sampler;
// Bumpmap is bound (INR texture slot 2) but unused: the 2D renderer has no
// lighting stage. Kept so the bind group layout carries the full part data.
@group(2) @binding(5) var tex_bumpmap: texture_2d<f32>;
@group(2) @binding(6) var samp_bumpmap: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Textures are straight alpha; the sRGB view decodes to linear on sample.
    // Premultiply here so blending stays premultiplied in linear space
    // (pipeline blend state is One / OneMinusSrcAlpha). Composite RTs were
    // rendered with that blend state, so they are already premultiplied.
    let albedo_tex = textureSample(tex_albedo, samp_albedo, in.uv);
    var albedo = vec4<f32>(albedo_tex.rgb * albedo_tex.a, albedo_tex.a);
    if material.composite != 0u {
        albedo = albedo_tex;
    }

    let screen_blend = material.screen_tint * albedo.a;
    let screen_out = vec3<f32>(1.0) - ((vec3<f32>(1.0) - albedo.rgb) * (vec3<f32>(1.0) - screen_blend));

    var color = vec4<f32>(screen_out * material.tint, albedo.a) * material.opacity;

    // Emission: additive on top of the shaded color, masked by its own alpha.
    // strength is 0 whenever the part has no emissive texture.
    let emissive = textureSample(tex_emissive, samp_emissive, in.uv);
    color = vec4<f32>(
        color.rgb + emissive.rgb * emissive.a * material.emissive_strength * material.opacity,
        color.a,
    );

    return color;
}
