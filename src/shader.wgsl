struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Vertex colors arrive already premultiplied from the MeshBuilder, matching
    // the pipeline's PREMULTIPLIED_ALPHA_BLENDING state. Interpolating them in
    // premultiplied form is also what makes the one-pixel feathered edges ramp
    // correctly: rgb and alpha fall to zero together, so an edge fades out
    // instead of darkening toward black on the way.
    //
    // A vertex with rgb > 0 and a == 0 therefore composites as dst + src, which
    // is how the glow layers accumulate additively without a second pipeline.
    return in.color;
}
