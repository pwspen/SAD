// WGSL assembly (originally ported from wgpu-py's sad_shaders.py).
// Fetches sad_shared.wgsl (the COMMON block between // BEGIN COMMON / // END COMMON)
// and concatenates it with each per-kernel snippet from shaders/wgsl/*.wgsl.
// gradients_tiled's `const TILE_SLOTS` line is rewritten at load time.

// Channels per site in the interleaved gradient buffer (see gradients_tiled.wgsl).
export const GRAD_CHANNELS = 10;

// Workgroup bytes per tile slot: key + 10 lo + 10 hi grad words.
const TILE_SLOT_BYTES = 21 * 4;

// Largest power-of-two tile table (128 or 256) that fits the device's
// workgroup storage. Pass the device's maxComputeWorkgroupStorageSize.
export function tileSlotsForLimit(maxWorkgroupStorage = 16384) {
  return maxWorkgroupStorage >= 256 * TILE_SLOT_BYTES ? 256 : 128;
}

const TILE_SLOTS_RE = /const TILE_SLOTS : u32 = \d+u;/;

function withTileSlots(source, tileSlots) {
  if (!TILE_SLOTS_RE.test(source)) {
    throw new Error("gradients_tiled.wgsl is missing its `const TILE_SLOTS : u32 = Nu;` line");
  }
  return source.replace(TILE_SLOTS_RE, `const TILE_SLOTS : u32 = ${tileSlots}u;`);
}

const COMMON_URL = new URL("../shaders/sad_shared.wgsl", import.meta.url);
const WGSL_DIR_URL = new URL("../shaders/wgsl/", import.meta.url);

const SHADER_FILES = [
  "init_gradient",
  "init_cand",
  "seed_cand",
  "vpt",
  "candidate_pack",
  "jfa_clear",
  "jfa_flood",
  "render_compute",
  "gradients_tiled",
  "adam",
  "clear_buffer_u32",
  "clear_buffer_i32",
  "tau_extract",
  "tau_diffuse",
  "tau_writeback",
  "hilbert",
  "radix_sort",
  "render_hashed",
  "render_tau_heatmap",
];

let _cache = null;

// Default reader for the browser. Node passes its own `readText` (fs-backed),
// since fetch() can't read file: URLs there.
async function fetchText(url) {
  // Cache-bust so local shader edits always take effect on reload.
  const bust = new URL(url);
  bust.searchParams.set("v", Date.now().toString());
  const response = await fetch(bust, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return await response.text();
}

function extractCommon(source) {
  const beginMarker = "// BEGIN COMMON";
  const endMarker = "// END COMMON";
  const start = source.indexOf(beginMarker);
  const end = source.indexOf(endMarker);
  if (start === -1 || end === -1) {
    throw new Error("Missing COMMON markers in sad_shared.wgsl");
  }
  return source.slice(start + beginMarker.length, end).trim() + "\n";
}

// tileSlots: gradient tile hash size (see tileSlotsForLimit).
// readText: async (URL) => string; defaults to fetch().
export async function loadShaders({ tileSlots = 128, readText = fetchText } = {}) {
  if (_cache && _cache.tileSlots === tileSlots) return _cache;

  const commonSource = await readText(COMMON_URL);
  const common = extractCommon(commonSource);

  const snippets = await Promise.all(
    SHADER_FILES.map((name) =>
      readText(new URL(`${name}.wgsl`, WGSL_DIR_URL)).then((text) => [name, text])
    )
  );

  const out = { COMMON: common };
  for (const [name, body] of snippets) {
    out[name] = common + body;
  }

  _cache = {
    INIT_GRADIENT: out.init_gradient,
    INIT_CAND: out.init_cand,
    SEED_CAND: out.seed_cand,
    VPT: out.vpt,
    CANDIDATE_PACK: out.candidate_pack,
    JFA_CLEAR: out.jfa_clear,
    JFA_FLOOD: out.jfa_flood,
    RENDER: out.render_compute,
    GRADIENTS_TILED: withTileSlots(out.gradients_tiled, tileSlots),
    ADAM: out.adam,
    CLEAR_BUFFER_U32: out.clear_buffer_u32,
    CLEAR_BUFFER_I32: out.clear_buffer_i32,
    TAU_EXTRACT: out.tau_extract,
    TAU_DIFFUSE: out.tau_diffuse,
    TAU_WRITEBACK: out.tau_writeback,
    HILBERT: out.hilbert,
    RADIX_SORT: out.radix_sort,
    RENDER_HASHED: out.render_hashed,
    RENDER_TAU_HEATMAP: out.render_tau_heatmap,
    tileSlots,
  };
  return _cache;
}
