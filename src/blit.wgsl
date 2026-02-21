// Blit shader: SceneFramebuffer (Rgba8Unorm, sRGB data) => ViewTarget (Rgba8UnormSrgb)
//
// El scene buffer contiene datos sRGB sin conversion
// El ViewTarget es Rgba8UnormSrgb: el hardware aplica linear sRGB al escribir
// Para que el resultado sea correcto: sRGB linear (shader) => sRGB (hardware)

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// Exact sRGB => linear conversion
fn srgb_to_linear_channel(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_channel(c.r),
        srgb_to_linear_channel(c.g),
        srgb_to_linear_channel(c.b),
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(tex, tex_sampler, in.uv);

    if (color.a < 0.001) {
        return vec4(0.0);
    }

    // Scene buffer tiene premultiplied alpha en sRGB space:
    //  stored = (R_srgb * A, G_srgb * A, B_srgb * A, A)
    //
    // Para escribir al ViewTarget (Rgba8UnormSrgb) con blend premultiplied:
    // Un-premultiply para obtener sRGB straight
    // sRGB => linear
    // Re-premultiply en linear space
    // Hardware aplica linear => sRGB al escribir

    let straight_srgb = color.rgb / color.a;
    let linear = srgb_to_linear(straight_srgb);
    return vec4(linear * color.a, color.a);
}
