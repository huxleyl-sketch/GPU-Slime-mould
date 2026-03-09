# WebGPU Plain JS Starter

A minimal WebGPU project using plain JavaScript and annotated WGSL.

## Run
WebGPU requires HTTPS or localhost. Start a local server in this folder:

```sh
python3 -m http.server 5173
```

Then open `http://localhost:5173`.

## Files
- `index.html` - static page with a canvas.
- `main.js` - WebGPU setup; runs a compute pass then renders.
- `shader.wgsl` - Render shader (WGSL) that reads from storage.
- `compute.wgsl` - Compute shader (WGSL) that writes colors.
- `style.css` - simple styling.

## Next ideas
- Add a uniform buffer to animate color.
- Add a vertex buffer instead of hardcoded arrays.
- Add a depth buffer and draw multiple triangles.
