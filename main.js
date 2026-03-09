const statusEl = document.querySelector('#status');
const canvas = document.querySelector('#gfx');
const saSlider = document.querySelector('#sa');
const raSlider = document.querySelector('#ra');
const soSlider = document.querySelector('#so');
const decaySlider = document.querySelector('#decay');
const saValue = document.querySelector('#sa-value');
const raValue = document.querySelector('#ra-value');
const soValue = document.querySelector('#so-value');
const decayValue = document.querySelector('#decay-value');

function setStatus(text) {
  statusEl.textContent = text;
}

if (!('gpu' in navigator)) {
  setStatus('WebGPU not supported in this browser. Try Chrome/Edge on desktop.');
  throw new Error('WebGPU not supported');
}

async function init() {
  setStatus('Requesting GPU adapter…');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    setStatus('No suitable GPU adapter found.');
    return;
  }

  setStatus('Requesting GPU device…');
  const device = await adapter.requestDevice();

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  });

  const [renderSource, computeSource] = await Promise.all([
    fetch('./shader.wgsl'),
    fetch('./compute.wgsl'),
  ]);
  if (!renderSource.ok) {
    throw new Error(`Failed to load shader.wgsl: ${renderSource.status}`);
  }
  if (!computeSource.ok) {
    throw new Error(`Failed to load compute.wgsl: ${computeSource.status}`);
  }
  const [renderCode, computeCode] = await Promise.all([
    renderSource.text(),
    computeSource.text(),
  ]);

  const renderModule = device.createShaderModule({ code: renderCode });
  const computeModule = device.createShaderModule({ code: computeCode });

  const SCENT_WIDTH = 1000;
  const SCENT_HEIGHT = 1000;
  const AGENT_COUNT = 50000;

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width));
    canvas.height = Math.max(1, Math.floor(rect.height));
    context.configure({
      device,
      format,
      alphaMode: 'opaque',
    });
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  const agentsData = new Float32Array(AGENT_COUNT * 4);
  for (let i = 0; i < AGENT_COUNT; i += 1) {
    const base = i * 4;
    agentsData[base + 0] = Math.random() * SCENT_WIDTH;
    agentsData[base + 1] = Math.random() * SCENT_HEIGHT;
    agentsData[base + 2] = Math.random() * Math.PI * 2;
    agentsData[base + 3] = 0;
  }

  const agentsBuffer = device.createBuffer({
    size: agentsData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(agentsBuffer, 0, agentsData);

  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  function createScentTexture() {
    return device.createTexture({
      size: { width: SCENT_WIDTH, height: SCENT_HEIGHT },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });
  }

  let scentRead = createScentTexture();
  let scentWrite = createScentTexture();

  const empty = new Uint8Array(SCENT_WIDTH * SCENT_HEIGHT * 4);
  device.queue.writeTexture(
    { texture: scentRead },
    empty,
    { bytesPerRow: SCENT_WIDTH * 4 },
    { width: SCENT_WIDTH, height: SCENT_HEIGHT }
  );
  device.queue.writeTexture(
    { texture: scentWrite },
    empty,
    { bytesPerRow: SCENT_WIDTH * 4 },
    { width: SCENT_WIDTH, height: SCENT_HEIGHT }
  );

  const agentsBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const diffuseBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: 'unfilterable-float' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: 'write-only', format: 'rgba8unorm' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const computeAgentsPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [agentsBindGroupLayout] }),
    compute: {
      module: computeModule,
      entryPoint: 'cs_agents',
    },
  });

  const computeDiffusePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [diffuseBindGroupLayout] }),
    compute: {
      module: computeModule,
      entryPoint: 'cs_defuse',
    },
  });

  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderModule,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: renderModule,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  const agentWorkgroups = Math.ceil(AGENT_COUNT / 256);
  const diffuseWorkgroupsX = Math.ceil(SCENT_WIDTH / 8);
  const diffuseWorkgroupsY = Math.ceil(SCENT_HEIGHT / 8);

  function makeAgentsBindGroup(inputTexture, outputTexture) {
    return device.createBindGroup({
      layout: agentsBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: agentsBuffer },
        },
        {
          binding: 1,
          resource: inputTexture.createView(),
        },
        {
          binding: 2,
          resource: outputTexture.createView(),
        },
        {
          binding: 3,
          resource: { buffer: paramsBuffer },
        },
      ],
    });
  }

  function makeDiffuseBindGroup(inputTexture, outputTexture) {
    return device.createBindGroup({
      layout: diffuseBindGroupLayout,
      entries: [
        {
          binding: 1,
          resource: inputTexture.createView(),
        },
        {
          binding: 2,
          resource: outputTexture.createView(),
        },
        {
          binding: 3,
          resource: { buffer: paramsBuffer },
        },
      ],
    });
  }

  function makeRenderBindGroup(texture) {
    return device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: texture.createView(),
        },
        {
          binding: 1,
          resource: sampler,
        },
      ],
    });
  }

  function angleFromSlider(t) {
    const low = 0.5;
    const high = 1.2;
    const max = Math.PI * 2;
    // Allocate most slider resolution to [0.5, 1.2].
    if (t < 0.7) {
      const u = t / 0.7;
      return low * u;
    }
    if (t < 0.9) {
      const u = (t - 0.7) / 0.2;
      return low + (high - low) * u;
    }
    const u = (t - 0.9) / 0.1;
    return high + (max - high) * u;
  }

  function updateParams() {
    const saT = Number.parseFloat(saSlider.value);
    const raT = Number.parseFloat(raSlider.value);
    const sa = angleFromSlider(saT);
    const ra = angleFromSlider(raT);
    const so = Number.parseFloat(soSlider.value);
    const decayT = Number.parseFloat(decaySlider.value);
    const decayMin = 0.5;
    const decayMax = 1.0;
    const decayK = 9.0;
    const decay =
      decayMin +
      (decayMax - decayMin) *
        (Math.log1p(decayK * decayT) / Math.log1p(decayK));
    saValue.textContent = sa.toFixed(2);
    raValue.textContent = ra.toFixed(2);
    soValue.textContent = so.toFixed(0);
    decayValue.textContent = decay.toFixed(2);
    device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([sa, ra, so, decay]));
  }

  saSlider.addEventListener('input', updateParams);
  raSlider.addEventListener('input', updateParams);
  soSlider.addEventListener('input', updateParams);
  decaySlider.addEventListener('input', updateParams);
  updateParams();

  function frame() {
    const encoder = device.createCommandEncoder();

    const agentsBindGroup = makeAgentsBindGroup(scentRead, scentWrite);
    const diffuseBindGroup = makeDiffuseBindGroup(scentWrite, scentRead);

    const computePassA = encoder.beginComputePass();
    computePassA.setPipeline(computeAgentsPipeline);
    computePassA.setBindGroup(0, agentsBindGroup);
    computePassA.dispatchWorkgroups(agentWorkgroups);
    computePassA.end();

    const computePassB = encoder.beginComputePass();
    computePassB.setPipeline(computeDiffusePipeline);
    computePassB.setBindGroup(0, diffuseBindGroup);
    computePassB.dispatchWorkgroups(diffuseWorkgroupsX, diffuseWorkgroupsY);
    computePassB.end();

    const renderBindGroup = makeRenderBindGroup(scentRead);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, renderBindGroup);
    pass.draw(3, 1, 0, 0);
    pass.end();

    device.queue.submit([encoder.finish()]);

    // Ping-pong the textures so the next frame reads the newly diffused scent.
    [scentRead, scentWrite] = [scentWrite, scentRead];
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
  setStatus('Running slime mould compute + render.');
}

init().catch((err) => {
  console.error(err);
  setStatus('Failed to initialize WebGPU. See console for details.');
});
