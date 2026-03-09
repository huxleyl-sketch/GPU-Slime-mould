//Slime Mould Formatting

struct Agent {
  //position as a 2d float vector
  pos : vec2f,
  //direction as a u32 float scalar
  dir : f32,
  //padding to ensure data isn't overwritten
  _pad : f32,
}

const agentCount : u32 = 50000;

//Stage Attributes
const height : u32 = 1000;
const width : u32 = 1000;


struct Params {
  sa : f32,
  ra : f32,
  so : f32,
  decay : f32,
};

@group(0) @binding(0) var<storage, read_write> agents : array<Agent>;
@group(0) @binding(1) var scentIn: texture_2d<f32>;
@group(0) @binding(2) var scentOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params : Params;

//Deposition

fn sense_at(pos : vec2f, dir : f32, sensorAngle : f32, sensorOffset : f32, texSize : vec2f) -> f32 {
  let sDir = dir + sensorAngle;
  let offset = vec2f(cos(sDir), sin(sDir)) * sensorOffset;
  let sPos = pos + offset;

  // Clamp to texture bounds
  let clamped = clamp(sPos, vec2f(0.0, 0.0), texSize - vec2f(1.0, 1.0));
  let coord = vec2i(clamped);

  // Read scent value (assumes single channel stored in .r)
  return textureLoad(scentIn, coord, 0).r;
}

fn sensor_stage(pos : vec2f, dir : f32, texSize : vec2f) -> vec3f {
  let left = sense_at(pos, dir, -params.sa, params.so, texSize);
  let center = sense_at(pos, dir, 0.0, params.so, texSize);
  let right = sense_at(pos, dir, params.sa, params.so, texSize);
  return vec3f(left, center, right);
}




@compute @workgroup_size(256)
fn cs_agents(@builtin(global_invocation_id) id : vec3u) {
  let i = id.x;                 // current agent index
  if (i >= agentCount) { return; }   

  let sensors = sensor_stage(agents[i].pos, agents[i].dir, vec2f(textureDimensions(scentIn)));

  // write updated agent
  if (
    sensors[0] > sensors[1] &&
    sensors[0] > sensors[2]
  ){
    //L >>
    agents[i].dir = agents[i].dir - params.ra;
  }
  else if (sensors[2] > sensors[1]){
    //R > M
    //R >>
    agents[i].dir = agents[i].dir + params.ra;
  }
  else { /*M >> */ }
  //Move in agent.dir
  let dirVec = vec2f(cos(agents[i].dir), sin(agents[i].dir));
  let dims = vec2f(textureDimensions(scentIn));
  var nextPos = agents[i].pos + dirVec;

  // wrap: ((x % w) + w) % w to handle negatives
  nextPos = vec2f(
    (nextPos.x % dims.x + dims.x) % dims.x,
    (nextPos.y % dims.y + dims.y) % dims.y
  );
  agents[i].pos = nextPos;
  
  let coord = vec2i(agents[i].pos);

  textureStore(scentOut, coord, vec4f(1.0, 0.0, 0.0, 1.0));
}
@compute @workgroup_size(8, 8)
fn cs_defuse(@builtin(global_invocation_id) id : vec3u) {
  let x = id.x;
  let y = id.y;
  if (x >= width || y >= height) { return; }

  let coord = vec2i(i32(x), i32(y));
  let newValue = blur3x3(scentIn,coord);
  
  textureStore(scentOut, coord, vec4f(newValue * params.decay, 0.0, 0.0, 1.0)); // write pixel
}
const KERNEL : array<f32, 9> = array<f32, 9>(
  1.0, 2.0, 1.0,
  2.0, 4.0, 2.0,
  1.0, 2.0, 1.0
);
const KERNEL_SUM : f32 = 16.0;

fn sample_clamped(tex: texture_2d<f32>, coord: vec2i) -> f32 {
  let dims = vec2i(textureDimensions(tex));
  let c = clamp(coord, vec2i(0, 0), dims - vec2i(1, 1));
  return textureLoad(tex, c, 0).r;
}

fn blur3x3(tex: texture_2d<f32>, coord: vec2i) -> f32 {
  var acc: f32 = 0.0;
  var k: u32 = 0u;
  for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
      let w = KERNEL[k];
      acc = acc + w * sample_clamped(tex, coord + vec2i(ox, oy));
      k = k + 1u;
    }
  }
  return acc / KERNEL_SUM;
}
