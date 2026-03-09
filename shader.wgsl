// WGSL render shader: draw a full-screen triangle and sample the scent texture.

struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@group(0) @binding(0)
var scentTex : texture_2d<f32>;

@group(0) @binding(1)
var scentSampler : sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var uvs = array<vec2f, 3>(
    vec2f(0.0, 0.0),
    vec2f(2.0, 0.0),
    vec2f(0.0, 2.0)
  );

  var out : VertexOut;
  out.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  out.uv = uvs[vertexIndex];
  return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4f {
  let scent = textureSample(scentTex, scentSampler, in.uv).r;
  let v = clamp(scent * 8.0, 0.0, 1.0);
  return vec4f(v, v, v, 1.0);
}
