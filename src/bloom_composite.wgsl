// Final bloom pass: adds the blurred bright-pass back onto the full-resolution
// scene and writes straight to the swapchain. A separate module from
// bloom_sample.wgsl (rather than more entry points in one file) because it
// binds a different resource set — two textures instead of one — and WGSL
// resource bindings are validated per module, so sharing group/binding
// numbers across unrelated layouts in the same file invites exactly the kind
// of mismatch this split avoids.

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VsOut;
    out.clip_position = vec4<f32>(pos[idx], 0.0, 1.0);
    out.uv = uv[idx];
    return out;
}

struct Params {
    threshold: f32,
    intensity: f32,
    dir: vec2<f32>,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(scene_tex, tex_sampler, in.uv).rgb;
    // bloom_tex is half resolution; the sampler's linear filtering upscales
    // it for free on the way in.
    let bloom = textureSample(bloom_tex, tex_sampler, in.uv).rgb;
    let color = scene + bloom * params.intensity;
    // The target is an 8-bit UNORM surface, not float — values past 1.0
    // would wrap instead of clip, so the clamp is load-bearing, not defensive.
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
