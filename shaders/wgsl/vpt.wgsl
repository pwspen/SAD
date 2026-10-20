@group(0) @binding(0) var<storage, read> sites : array<PackedCandidateSite>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var inCand0 : texture_2d<u32>;
@group(0) @binding(3) var inCand1 : texture_2d<u32>;
@group(0) @binding(4) var outCand0 : texture_storage_2d<rgba32uint, write>;
@group(0) @binding(5) var outCand1 : texture_storage_2d<rgba32uint, write>;

// The tile of the candidate pixel being updated (Tiling; 0 untiled), and its
// top-left pixel.
var<private> pixTile : u32;
var<private> pixOrigin : vec2<f32>;

fn tryInsert8(bestIdx : ptr<function, array<u32, 8>>, bestD2 : ptr<function, array<f32, 8>>,
              candIdx : u32, uv : vec2<f32>, siteCount : u32, invScaleSq : f32, inv_scale : f32) {
  // The global site is blended separately (cand_ids), never stored; a tile
  // only lists its own sites.
  if (candIdx >= siteCount || candIdx == tile_global(params.tiling, pixTile, params.globalCand) ||
      !site_in_tile(params.tiling, candIdx, pixTile)) { return; }
  for (var i = 0u; i < 8u; i = i + 1u) {
    if ((*bestIdx)[i] == candIdx) { return; }
  }
  let packed = sites[candIdx];
  // Tiled, packed positions are tile-relative (candidate_pack.wgsl); the
  // site is in the pixel's tile, so offset it by that tile's origin.
  let pos = unpack2x16float(packed.data.x) + pixOrigin;
  let tau_rad = unpack2x16float(packed.data.y);
  let dir = unpack2x16float(packed.data.z);
  let log_aniso = unpack2x16float(packed.data.w).x;
  var d2 : f32;
  if (pos.x < 0.0 || is_bad2(pos) || is_bad(tau_rad.x) || is_bad(log_aniso) || is_bad2(dir)) {
    return;
  } else {
    let diff = uv - pos;
    let diff2 = dot(diff, diff);
    let proj = dot(dir, diff);
    let proj2 = proj * proj;
    let perp2 = max(diff2 - proj2, 0.0);

    let l1 = exp(log_aniso);
    let l2 = 1.0 / max(l1, 1e-8);
    let d2_aniso = l1 * proj2 + l2 * perp2;
    let d2_norm = d2_aniso * invScaleSq;
    let d2_safe = max(d2_norm, 1e-8);
    let r_norm = tau_rad.y; // packed normalized (candidate_pack.wgsl)
    let dmix2 = sqrt(d2_safe) - r_norm;
    if (is_bad(dmix2)) { return; }
    let tau = max(exp(tau_rad.x), 1e-4);
    if (is_bad(tau)) { return; }
    d2 = tau * dmix2;
    if (is_bad(d2)) { return; }
  }
  if (d2 >= (*bestD2)[7]) { return; }
  var insertPos = 7u;
  for (var i = 0u; i < 8u; i = i + 1u) {
    if (d2 < (*bestD2)[i]) { insertPos = i; break; }
  }
  for (var i = 7u; i > insertPos; i = i - 1u) {
    (*bestD2)[i] = (*bestD2)[i - 1u];
    (*bestIdx)[i] = (*bestIdx)[i - 1u];
  }
  (*bestD2)[insertPos] = d2;
  (*bestIdx)[insertPos] = candIdx;
}

// Offers all 8 ids of a candidate texel pair to the list.
fn tryInsertTexel(bestIdx : ptr<function, array<u32, 8>>, bestD2 : ptr<function, array<f32, 8>>,
                  c0 : vec4<u32>, c1 : vec4<u32>, uv : vec2<f32>, siteCount : u32,
                  invScaleSq : f32, inv_scale : f32) {
  let ids = cand_ids8(c0, c1);
  for (var i = 0u; i < 8u; i = i + 1u) {
    tryInsert8(bestIdx, bestD2, ids[i], uv, siteCount, invScaleSq, inv_scale);
  }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let candW = max(params.candWidth, 1u);
  let candH = max(params.candHeight, 1u);
  if (gid.x >= candW || gid.y >= candH) { return; }

  let stepIndex = params.step & 0xffffu;
  let jumpStep = max(1u, params.step >> 16u);
  let fullStepIndex = (params.seed << 16u) | stepIndex;

  let uv = candidateUvFromCoord(vec2<u32>(gid.xy), vec2<u32>(params.width, params.height), vec2<u32>(candW, candH));
  pixTile = tile_of_pixel(params.tiling, vec2<u32>(uv));
  if (tiled(params.tiling)) { pixOrigin = site_tile_origin(params.tiling, pixTile * params.tiling.sitesPerTile); }
  let gi = vec2<i32>(gid.xy);
  let w = i32(candW);
  let h = i32(candH);

  let inv_scale = sqrt(params.invScaleSq);

  var bestIdx : array<u32, 8>;
  let inf = 1e30;
  var bestD2 : array<f32, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    bestIdx[i] = 0xffffffffu;
    bestD2[i] = inf;
  }

  tryInsertTexel(&bestIdx, &bestD2, textureLoad(inCand0, gi, 0), textureLoad(inCand1, gi, 0),
                 uv, params.siteCount, params.invScaleSq, inv_scale);

  let offsets = array<vec2<i32>, 4>(
    vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1)
  );

  for (var i = 0u; i < 4u; i = i + 1u) {
    let p = clamp(gi + offsets[i] * i32(jumpStep), vec2<i32>(0), vec2<i32>(w - 1, h - 1));
    tryInsertTexel(&bestIdx, &bestD2, textureLoad(inCand0, p, 0), textureLoad(inCand1, p, 0),
                   uv, params.siteCount, params.invScaleSq, inv_scale);
  }

  // Probe distance scales with the image: the tile's width when tiled.
  var gridW = f32(candW);
  if (tiled(params.tiling)) { gridW = f32(params.tiling.tileW) / f32(max(params.candDownscale, 1u)); }
  let rad = params.radiusScale * (gridW / 1024.0);
  var state = (gid.x * 73856093u) ^ (gid.y * 19349663u) ^ ((stepIndex + jumpStep) * 83492791u);
  for (var r = 0u; r < params.radiusProbes; r = r + 1u) {
    let a = rand01(&state) * 6.2831853;
    let dx = i32(cos(a) * rad);
    let dy = i32(sin(a) * rad);
    let p = clamp(gi + vec2<i32>(dx, dy), vec2<i32>(0), vec2<i32>(w - 1, h - 1));
    if (gid.x == u32(p.x) && gid.y == u32(p.y)) { continue; }
    tryInsertTexel(&bestIdx, &bestD2, textureLoad(inCand0, p, 0), textureLoad(inCand1, p, 0),
                   uv, params.siteCount, params.invScaleSq, inv_scale);
  }

  var injectState = (gid.x * 1664525u) ^ (gid.y * 1013904223u) ^ (fullStepIndex * 374761393u);
  for (var i = 0u; i < params.injectCount; i = i + 1u) {
    injectState = xorshift32(injectState + i);
    var cand = injectState % params.siteCount;
    if (tiled(params.tiling)) {
      cand = pixTile * params.tiling.sitesPerTile + injectState % params.tiling.sitesPerTile;
    }
    tryInsert8(&bestIdx, &bestD2, cand, uv, params.siteCount, params.invScaleSq, inv_scale);
  }

  textureStore(outCand0, gi, vec4<u32>(bestIdx[0], bestIdx[1], bestIdx[2], bestIdx[3]));
  textureStore(outCand1, gi, vec4<u32>(bestIdx[4], bestIdx[5], bestIdx[6], bestIdx[7]));
}
