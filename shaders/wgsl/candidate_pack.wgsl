@group(0) @binding(0) var<storage, read> sites : array<Site>;
@group(0) @binding(1) var<storage, read_write> packed_sites : array<PackedCandidateSite>;
// invScale = 1 / max(width, height) (a tile's when tiled). The radius is
// packed normalized (r · invScale, as the logit uses it): in pixels a wide flat
// site's radius overflows a half float. Tiled, positions are packed relative to
// the site's tile, so half-float precision doesn't depend on the atlas size.
struct PackParams {
  count : u32,
  invScale : f32,
  _pad0 : u32,
  _pad1 : u32,
  @align(16) tiling : Tiling,
}
@group(0) @binding(2) var<uniform> packParams : PackParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= packParams.count) { return; }

  let site = sites[idx];
  var out : PackedCandidateSite;
  if (site.position.x < 0.0) {
    out.data = vec4<u32>(
      pack2x16float(vec2<f32>(-1.0, -1.0)),
      pack2x16float(vec2<f32>(0.0, 0.0)),
      pack2x16float(vec2<f32>(0.0, 0.0)),
      pack2x16float(vec2<f32>(0.0, 0.0))
    );
  } else {
    out.data = vec4<u32>(
      pack2x16float(site.position - select(vec2<f32>(0.0), site_tile_origin(packParams.tiling, idx),
                                           tiled(packParams.tiling))),
      pack2x16float(vec2<f32>(site.log_tau, site.radius_sq * packParams.invScale)),
      pack2x16float(site_aniso_dir(site)),
      pack2x16float(vec2<f32>(site.log_aniso, 0.0))
    );
  }
  packed_sites[idx] = out;
}
