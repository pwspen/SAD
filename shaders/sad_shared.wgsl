

struct Site {
  position : vec2<f32>,
  log_tau : f32,
  radius_sq : f32,
  // Color params in COLOR_SPACE (plain RGB by default); read via site_color().
  color_r : f32,
  color_g : f32,
  color_b : f32,
  aniso_dir_x : f32,
  aniso_dir_y : f32,
  log_aniso : f32,
}

// Packed site format for candidate search (four half2 values).
struct PackedCandidateSite {
  data : vec4<u32>,
}

// Site color parameterization; see trainer/color_spaces.js, which must agree
// with the three functions below. color_r/g/b hold params in this space and
// site_color() decodes them to RGB. The trainer rewrites this line per run.
// 0 = RGB, 1 = log intensity + chroma (mean log RGB, log R/G, log B/G),
// 2 = YCoCg (real-valued, linear), 3 = KLT: coordinates along the basis
// below (columns = orthonormal axes, about the mean), which the trainer
// rewrites per run like COLOR_SPACE.
const COLOR_SPACE : u32 = 0u;
const COLOR_BASIS_AXES : mat3x3<f32> = mat3x3<f32>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
const COLOR_BASIS_MEAN : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const LOG_CHROMA_EPS : f32 = 0.00392156863; // 1/255

fn decode_color(p : vec3<f32>) -> vec3<f32> {
  switch COLOR_SPACE {
    case 1u: {
      let lg = p.x - (p.y + p.z) / 3.0;
      return exp(vec3<f32>(lg + p.y, lg, lg + p.z)) - LOG_CHROMA_EPS;
    }
    case 2u: { return vec3<f32>(p.x - p.z + p.y, p.x + p.z, p.x - p.z - p.y); }
    case 3u: { return COLOR_BASIS_MEAN + COLOR_BASIS_AXES * p; }
    default: { return p; }
  }
}

fn encode_color(rgb : vec3<f32>) -> vec3<f32> {
  switch COLOR_SPACE {
    case 1u: {
      let l = log(max(rgb, vec3<f32>(0.0)) + LOG_CHROMA_EPS);
      return vec3<f32>((l.x + l.y + l.z) / 3.0, l.x - l.y, l.z - l.y);
    }
    case 2u: {
      return vec3<f32>((rgb.x + 2.0 * rgb.y + rgb.z) / 4.0, (rgb.x - rgb.z) / 2.0, (2.0 * rgb.y - rgb.x - rgb.z) / 4.0);
    }
    case 3u: { return (rgb - COLOR_BASIS_MEAN) * COLOR_BASIS_AXES; }
    default: { return rgb; }
  }
}

// dLoss/dparams from dLoss/dRGB at params p (vector-Jacobian product).
fn color_vjp(p : vec3<f32>, g_rgb : vec3<f32>) -> vec3<f32> {
  switch COLOR_SPACE {
    case 1u: {
      let lg = p.x - (p.y + p.z) / 3.0;
      let u = g_rgb * exp(vec3<f32>(lg + p.y, lg, lg + p.z));
      return vec3<f32>(u.x + u.y + u.z, (2.0 * u.x - u.y - u.z) / 3.0, (2.0 * u.z - u.y - u.x) / 3.0);
    }
    case 2u: { return vec3<f32>(g_rgb.x + g_rgb.y + g_rgb.z, g_rgb.x - g_rgb.z, g_rgb.y - g_rgb.x - g_rgb.z); }
    case 3u: { return g_rgb * COLOR_BASIS_AXES; }
    default: { return g_rgb; }
  }
}

fn site_color_params(site : Site) -> vec3<f32> {
  return vec3<f32>(site.color_r, site.color_g, site.color_b);
}

fn site_color(site : Site) -> vec3<f32> {
  return decode_color(site_color_params(site));
}

fn site_aniso_dir(site : Site) -> vec2<f32> {
  return vec2<f32>(site.aniso_dir_x, site.aniso_dir_y);
}

struct Params {
  width : u32,
  height : u32,
  siteCount : u32,
  step : u32,
  invScaleSq : f32,
  seed : u32,
  radiusScale : f32,
  radiusProbes : u32,
  injectCount : u32,
  candDownscale : u32,
  candWidth : u32,
  candHeight : u32,
  globalCand : u32, // the global site (cand_ids); none if >= siteCount
  @align(16) tiling : Tiling,
}

// Batched training (trainer/batch.js): the image is an atlas of tiles, one
// image each, tile b at column b % tilesX, row b / tilesX. Tile b owns sites
// [b·sitesPerTile, (b + 1)·sitesPerTile) and a site only ever blends inside
// its own tile. A global site id (the histogram layer) is then a slot within
// each tile. sitesPerTile = 0: one image, no tiles.
struct Tiling {
  tileW : u32,
  tileH : u32,
  sitesPerTile : u32,
  tilesX : u32,
}

fn tiled(t : Tiling) -> bool { return t.sitesPerTile > 0u; }

// The tile holding pixel p (0 when untiled).
fn tile_of_pixel(t : Tiling, p : vec2<u32>) -> u32 {
  if (!tiled(t)) { return 0u; }
  return (p.y / t.tileH) * t.tilesX + p.x / t.tileW;
}

// Whether site `idx` may blend in tile `tile`.
fn site_in_tile(t : Tiling, idx : u32, tile : u32) -> bool {
  return !tiled(t) || idx / t.sitesPerTile == tile;
}

// The global site of `tile`: `global` itself untiled, else that slot of the tile.
fn tile_global(t : Tiling, tile : u32, global : u32) -> u32 {
  if (!tiled(t) || global == NO_SITE) { return global; }
  return tile * t.sitesPerTile + global;
}

// Top-left pixel of the tile holding site `idx`.
fn site_tile_origin(t : Tiling, idx : u32) -> vec2<f32> {
  let b = idx / t.sitesPerTile;
  return vec2<f32>(f32((b % t.tilesX) * t.tileW), f32((b / t.tilesX) * t.tileH));
}

struct InitParams {
  numSites : u32,
  gradThreshold : f32,
  maxAttempts : u32,
  skipSlot : u32, // tiled: the slot left for the histogram layer (none: NO_SITE)
  initLogTau : f32,
  initRadius : f32,
  _pad1 : u32,
  _pad2 : u32,
  @align(16) tiling : Tiling,
}

// ema*: the persistent-error weighting (gradients_tiled.wgsl). emaSlot is the
// errEmaSum slot holding last step's sum (this step adds to the other one),
// emaKeep the EWMA's decay this step, emaInvCount 1 / masked pixel count.
struct GradParams {
  siteCount : u32,
  globalCand : u32,
  emaSlot : u32,
  _pad0 : u32,
  invScaleSq : f32,
  emaKeep : f32,
  emaAlpha : f32,
  emaCap : f32,
  emaInvCount : f32,
  _pad1 : f32,
  _pad2 : f32,
  _pad3 : f32,
  @align(16) tiling : Tiling,
}

struct AdamParams {
  lr_pos : f32,
  lr_tau : f32,
  lr_radius : f32,
  lr_color : f32,
  lr_dir : f32,
  lr_aniso : f32,
  beta1 : f32,
  beta2 : f32,
  eps : f32,
  t : u32,
  width : u32,
  height : u32,
  layer_site : u32, // the histogram layer's index (trainer.js), or none; a slot per tile when tiled
  color_moment : u32, // 0: a second moment per color param; 1: one shared (trainer.js, COLOR_MOMENTS)
  _pad1 : u32,
  _pad2 : u32,
  // Tiled: width / height above are the tile's, and positions clamp to the
  // site's own tile.
  @align(16) tiling : Tiling,
}

struct ClearParams {
  count : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}

// RNG
fn xorshift32(x : u32) -> u32 {
  var s = x;
  s = s ^ (s << 13u);
  s = s ^ (s >> 17u);
  s = s ^ (s << 5u);
  return s;
}

fn rand01(state : ptr<function, u32>) -> f32 {
  *state = xorshift32(*state);
  return f32(*state) * (1.0 / 4294967296.0);
}

fn is_bad(x : f32) -> bool {
  return (x != x) || abs(x) > 1.0e30;
}

fn is_bad2(v : vec2<f32>) -> bool {
  return is_bad(v.x) || is_bad(v.y);
}

fn is_bad3(v : vec3<f32>) -> bool {
  return is_bad(v.x) || is_bad(v.y) || is_bad(v.z);
}

fn safe_dir(v : vec2<f32>) -> vec2<f32> {
  if (is_bad2(v)) { return vec2<f32>(1.0, 0.0); }
  let len2 = dot(v, v);
  if (len2 < 1.0e-12) { return vec2<f32>(1.0, 0.0); }
  return v * inverseSqrt(len2);
}

fn candidateCoord(gid : vec2<u32>, outputSize : vec2<u32>, candSize : vec2<u32>) -> vec2<u32> {
  let outW = max(outputSize.x, 1u);
  let outH = max(outputSize.y, 1u);
  let candW = max(candSize.x, 1u);
  let candH = max(candSize.y, 1u);
  let cx = min((gid.x * candW) / outW, candW - 1u);
  let cy = min((gid.y * candH) / outH, candH - 1u);
  return vec2<u32>(cx, cy);
}

fn candidateUvFromCoord(coord : vec2<u32>, outputSize : vec2<u32>, candSize : vec2<u32>) -> vec2<f32> {
  let outW = max(outputSize.x, 1u);
  let outH = max(outputSize.y, 1u);
  let candW = max(candSize.x, 1u);
  let candH = max(candSize.y, 1u);
  let outSizeF = vec2<f32>(f32(outW), f32(outH));
  let candSizeF = vec2<f32>(f32(candW), f32(candH));
  var uv = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5, 0.5)) * (outSizeF / candSizeF);
  uv = min(uv, outSizeF - vec2<f32>(1.0, 1.0));
  return uv;
}

fn hashColor(siteId : u32) -> vec3<f32> {
  var h = siteId;
  h = (h ^ 61u) ^ (h >> 16u);
  h = h + (h << 3u);
  h = h ^ (h >> 4u);
  let r = h * 0x27d4eb2du;

  h = siteId * 2654435761u;
  h = (h ^ 61u) ^ (h >> 16u);
  h = h + (h << 3u);
  let g = h * 0x27d4eb2du;

  h = siteId * 1103515245u;
  h = (h ^ 61u) ^ (h >> 16u);
  h = h ^ (h >> 4u);
  let b = h * 0x27d4eb2du;

  return vec3<f32>(
    f32(r & 0xFFFFFFu) / f32(0xFFFFFFu),
    f32(g & 0xFFFFFFu) / f32(0xFFFFFFu),
    f32(b & 0xFFFFFFu) / f32(0xFFFFFFu)
  );
}

const K_GRAD_QUANT_SCALE : f32 = 1000000.0;
const K_GRAD_QUANT_SCALE_INV : f32 = 1.0 / 1000000.0;

fn voronoi_dmix2_fast(site : Site, uv : vec2<f32>, invScaleSq : f32, inv_scale : f32) -> f32 {
  let diff = uv - site.position;
  let diff2 = dot(diff, diff);
  let proj = dot(site_aniso_dir(site), diff);
  let proj2 = proj * proj;
  let perp2 = max(diff2 - proj2, 0.0);
  let l1 = exp(site.log_aniso);
  let l2 = 1.0 / l1;
  let d2_aniso = l1 * proj2 + l2 * perp2;
  let d2_norm = d2_aniso * invScaleSq;
  let d2_safe = max(d2_norm, 1.0e-8);
  let r_norm = site.radius_sq * inv_scale;
  return (sqrt(d2_safe) - r_norm);
}

const NEG_INF : f32 = -1e30;

// The 8 candidate site ids stored in a candidate texel pair (cand0, cand1).
fn cand_ids8(c0 : vec4<u32>, c1 : vec4<u32>) -> array<u32, 8> {
  return array<u32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);
}

// Sites a pixel blends: its 8 stored candidates plus the global site (the
// histogram layer, offered to every pixel without taking a list slot; none
// when >= siteCount). A stored copy of the global site is dropped.
const NCAND : u32 = 9u;
const NO_SITE : u32 = 0xffffffffu;
fn cand_ids(c0 : vec4<u32>, c1 : vec4<u32>, global : u32) -> array<u32, NCAND> {
  var ids = array<u32, NCAND>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, global);
  for (var i = 0u; i < 8u; i = i + 1u) { if (ids[i] == global) { ids[i] = NO_SITE; } }
  return ids;
}

// A site's blend logit at pixel uv, -tau * (d - r) (voronoi_dmix2_fast), or
// NEG_INF for an inactive site or non-finite parameters. Every kernel that
// blends sites (render, gradients, site stats, info) goes through this and
// softmax8, so they agree on the weights.
fn site_logit(site : Site, uv : vec2<f32>, invScaleSq : f32, inv_scale : f32) -> f32 {
  if (site.position.x < 0.0 || is_bad(site.position.x) || is_bad(site.position.y) ||
      is_bad(site.log_tau) || is_bad(site.log_aniso)) { return NEG_INF; }
  let tau = exp(site.log_tau);
  if (is_bad(tau)) { return NEG_INF; }
  let dmix2 = voronoi_dmix2_fast(site, uv, invScaleSq, inv_scale);
  if (is_bad(dmix2)) { return NEG_INF; }
  let l = -tau * dmix2;
  if (is_bad(l)) { return NEG_INF; }
  return l;
}

// Normalized softmax weights of the NCAND logits (NEG_INF = no site). All
// zero when no logit is finite.
fn softmax_c(logits : array<f32, NCAND>) -> array<f32, NCAND> {
  var w : array<f32, NCAND>;
  var m = NEG_INF;
  for (var i = 0u; i < NCAND; i = i + 1u) { m = max(m, logits[i]); }
  if (m <= -1e29 || is_bad(m)) { return w; }
  var s = 0.0;
  for (var i = 0u; i < NCAND; i = i + 1u) {
    var e = exp(logits[i] - m);
    if (is_bad(e)) { e = 0.0; }
    w[i] = e;
    s = s + e;
  }
  if (is_bad(s) || !(s > 0.0)) { return array<f32, NCAND>(); }
  let inv = 1.0 / max(s, 1e-8);
  for (var i = 0u; i < NCAND; i = i + 1u) { w[i] = w[i] * inv; }
  return w;
}

