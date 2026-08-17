// Fullscreen threshold and separable-blur passes shared by the bloom chain.
// A fullscreen triangle (3 vertices, no vertex buffer) rather than a quad —
// one triangle covering the viewport avoids the diagonal seam a two-triangle
// quad would need to rasterize exactly, at the cost of overdraw outside the
// screen that the rasterizer clips away for free.

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

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;

// Bright-pass: subtract the threshold rather than a hard cutoff, so the glow
// fades in smoothly instead of snapping on at an edge. `intensity` is unused
// here but shares the struct with the composite pass's uniform layout.
@fragment
fn fs_threshold(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, src_sampler, in.uv).rgb;
    let bright = max(c - vec3<f32>(params.threshold), vec3<f32>(0.0));
    return vec4<f32>(bright, 1.0);
}

// 9-tap separable Gaussian (weights for a sigma ~2px kernel). `params.dir` is
// the per-axis texel step, so the same fragment shader does both the
// horizontal and vertical half of the blur depending on which uniform buffer
// is bound.
//
// Declared `var`, not module-scope `const`: naga rejects indexing a `const`
// array by a non-constant (loop-variable) index — "may only be indexed by a
// constant" — so the weights have to live in function scope as a `var`.
@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    var weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    var result = textureSample(src_tex, src_sampler, in.uv).rgb * weights[0];
    for (var i: i32 = 1; i < 5; i = i + 1) {
        let offset = params.dir * f32(i);
        result += textureSample(src_tex, src_sampler, in.uv + offset).rgb * weights[i];
        result += textureSample(src_tex, src_sampler, in.uv - offset).rgb * weights[i];
    }
    return vec4<f32>(result, 1.0);
}
