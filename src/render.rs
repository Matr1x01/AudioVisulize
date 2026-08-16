//! Drawing primitives and the visualizer interface.
//!
//! Geometry is emitted as indexed triangles into one `MeshBuilder`, drawn in a
//! single call with painter's-algorithm ordering (no depth buffer): later
//! geometry paints over earlier geometry.
//!
//! # Antialiasing model
//!
//! MSAA alone cannot make this look smooth. At 4x it resolves to four coverage
//! levels — two bits of edge gradient — which is visibly stepped on the
//! near-horizontal thin strokes this app is mostly made of. So every filled
//! edge here carries its own **1-pixel alpha-feathered fringe**: the outline is
//! extruded outward by one pixel to vertices whose alpha is zero, and the
//! hardware interpolator produces a smooth coverage ramp across it. MSAA then
//! cleans up whatever is left. The fringe is what actually removes the jaggies.
//!
//! # Color model
//!
//! Vertex colors are stored **premultiplied**, matching the pipeline's
//! `PREMULTIPLIED_ALPHA_BLENDING` state, and the fragment shader passes them
//! through untouched. Two consequences worth knowing:
//!
//! * A fringe vertex is `[0,0,0,0]` — transparent black — which contributes
//!   nothing under either blend mode.
//! * `rgb > 0` with `a == 0` composites as `dst + src`, i.e. **additive**. That
//!   is how the glow layers accumulate instead of occluding each other, with no
//!   second pipeline and no state changes.
//!
//! Callers pass ordinary straight-alpha colors; conversion happens here.

/// Minimum stroke width in pixels.
///
/// Below roughly one pixel a stroke's coverage varies with its sub-pixel
/// position, so a moving hairline shimmers and intermittently disappears.
/// Narrower requests are widened to this and their alpha scaled down by the
/// same factor, which keeps total emitted light constant — the line looks
/// fainter rather than broken.
pub const MIN_STROKE_PX: f32 = 1.25;

/// Width of the alpha ramp on every edge, in pixels.
///
/// One pixel is the natural choice: the ramp spans exactly one sample spacing,
/// so a fully-covered pixel stays fully covered and an uncovered one stays
/// clear, with a single pixel of gradient between. Wider looks blurry.
pub const FEATHER_PX: f32 = 1.0;

/// Miter length limit, as a multiple of half the stroke width.
///
/// A miter's length grows as `1/sin(theta/2)`, so it runs to infinity as a
/// corner closes up. 4.0 clamps that at about 29 degrees of included angle;
/// sharper than that the join is truncated instead, which is invisible in
/// practice because every curve here is spline-resampled and turns gently.
pub const MITER_LIMIT: f32 = 4.0;

/// Target arc length between subdivisions of a round cap, in pixels.
const CAP_SEGMENT_PX: f32 = 2.5;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    /// Premultiplied RGBA. See the module docs.
    pub color: [f32; 4],
}

/// Per-frame audio and timing state handed to every visualizer.
#[allow(dead_code)]
pub struct FrameData<'a> {
    /// Smoothed per-band levels, L/R averaged. Length is `NUM_BANDS`.
    pub bands: &'a [f32],
    pub left_bands: &'a [f32],
    pub right_bands: &'a [f32],
    /// Raw mono PCM for the current window (un-windowed).
    pub waveform: &'a [f32],
    pub bass: f32,
    pub rms: f32,
    /// Seconds since startup, for animation that isn't audio-driven.
    pub time: f32,
    /// Current background color, so visuals can fake occlusion by filling with it.
    pub background: [f32; 3],
}

/// One visualization style.
///
/// Implementations own whatever state they need across frames (history buffers,
/// peak holds, rotation angles, reusable point buffers) — they're constructed
/// once and reused, so nothing in `draw` needs to allocate.
pub trait Visualizer {
    fn name(&self) -> &'static str;
    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder);
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

/// Decode an sRGB hex literal (e.g. Material's `0x7C4DFF`) to linear RGB.
///
/// The surface is an sRGB format, so values written by the shader are treated
/// as linear light and encoded on store. Palette values published as hex are
/// sRGB-encoded, so they must be decoded here or every color comes out
/// washed-out and too bright.
pub fn srgb_hex(hex: u32) -> [f32; 3] {
    let ch = |shift: u32| {
        let s = ((hex >> shift) & 0xFF) as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    [ch(16), ch(8), ch(0)]
}

/// Linear interpolation between two linear-space colors.
pub fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Sample a piecewise-linear ramp of key colors at `t` in 0..1.
pub fn ramp(keys: &[[f32; 3]], t: f32) -> [f32; 3] {
    if keys.is_empty() {
        return [0.0; 3];
    }
    if keys.len() == 1 {
        return keys[0];
    }
    let t = t.clamp(0.0, 1.0) * (keys.len() - 1) as f32;
    let i = (t.floor() as usize).min(keys.len() - 2);
    mix(keys[i], keys[i + 1], t - i as f32)
}

pub fn rgba(rgb: [f32; 3], alpha: f32) -> [f32; 4] {
    [rgb[0], rgb[1], rgb[2], alpha]
}

// ---------------------------------------------------------------------------
// Splines
// ---------------------------------------------------------------------------

/// Resample a polyline through a **monotone cubic** (Fritsch–Carlson) spline.
///
/// Chosen over Catmull–Rom for band data specifically because it cannot
/// overshoot: a spectrum curve interpolated with an unconstrained cubic dips
/// below the baseline between a tall band and a silent one, which shows up as
/// the silhouette punching through its own floor. Fritsch–Carlson limits the
/// tangents so the curve stays monotone on every span where the data is,
/// removing that entirely.
///
/// `x` must be non-decreasing. `subdiv` is output points per input span.
/// Results are appended to `out`, which is cleared first; pass a buffer owned
/// by the visualizer so this never allocates.
pub fn resample_monotone(pts: &[[f32; 2]], subdiv: usize, out: &mut Vec<[f32; 2]>) {
    out.clear();
    if pts.len() < 2 {
        out.extend_from_slice(pts);
        return;
    }
    let n = pts.len();
    let subdiv = subdiv.max(1);

    // Secant slopes, and tangents as their averages.
    let mut slope = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = pts[i + 1][0] - pts[i][0];
        slope.push(if dx.abs() > 1e-9 {
            (pts[i + 1][1] - pts[i][1]) / dx
        } else {
            0.0
        });
    }

    let mut m = Vec::with_capacity(n);
    m.push(slope[0]);
    for i in 1..n - 1 {
        // A sign change means a local extremum; a zero tangent there is what
        // keeps the curve from overshooting past the data point.
        m.push(if slope[i - 1] * slope[i] <= 0.0 {
            0.0
        } else {
            (slope[i - 1] + slope[i]) * 0.5
        });
    }
    m.push(slope[n - 2]);

    // Fritsch–Carlson: clamp each tangent pair into the circle of radius 3
    // around the secant slope. Outside it the Hermite cubic is non-monotone.
    for i in 0..n - 1 {
        if slope[i].abs() < 1e-9 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let alpha = m[i] / slope[i];
        let beta = m[i + 1] / slope[i];
        let s = alpha * alpha + beta * beta;
        if s > 9.0 {
            let tau = 3.0 / s.sqrt();
            m[i] = tau * alpha * slope[i];
            m[i + 1] = tau * beta * slope[i];
        }
    }

    for i in 0..n - 1 {
        let (p0, p1) = (pts[i], pts[i + 1]);
        let h = p1[0] - p0[0];
        for k in 0..subdiv {
            let t = k as f32 / subdiv as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            // Cubic Hermite basis.
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            out.push([
                p0[0] + h * t,
                h00 * p0[1] + h10 * h * m[i] + h01 * p1[1] + h11 * h * m[i + 1],
            ]);
        }
    }
    out.push(pts[n - 1]);
}

// ---------------------------------------------------------------------------
// MeshBuilder
// ---------------------------------------------------------------------------

/// Accumulates indexed triangles in normalized device coordinates (-1..1).
///
/// Holds the framebuffer size so widths and radii can be specified in **pixels**
/// and converted here. Specifying them in NDC is what let thin strokes fall
/// below one pixel on short windows and start shimmering.
pub struct MeshBuilder {
    verts: Vec<Vertex>,
    indices: Vec<u32>,
    aspect: f32,
    width_px: f32,
    height_px: f32,
    /// Reused between strokes so tessellation never allocates per frame.
    ribs: Vec<[u32; 4]>,
    normals: Vec<[f32; 2]>,
}

/// Some helpers here are unused by the shipped visuals; they exist so new ones
/// have primitives to build from.
#[allow(dead_code)]
impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            verts: Vec::with_capacity(1 << 16),
            indices: Vec::with_capacity(1 << 17),
            aspect: 1.0,
            width_px: 1.0,
            height_px: 1.0,
            ribs: Vec::with_capacity(2048),
            normals: Vec::with_capacity(2048),
        }
    }

    pub fn begin(&mut self, width_px: f32, height_px: f32) {
        self.verts.clear();
        self.indices.clear();
        self.width_px = width_px.max(1.0);
        self.height_px = height_px.max(1.0);
        self.aspect = (self.width_px / self.height_px).max(1e-3);
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.verts
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn aspect(&self) -> f32 {
        self.aspect
    }

    pub fn size_px(&self) -> (f32, f32) {
        (self.width_px, self.height_px)
    }

    /// NDC-y units per pixel. The y axis spans 2.0 over `height_px`.
    pub fn px(&self) -> f32 {
        2.0 / self.height_px
    }

    /// Convert a length in NDC-y units to pixels.
    pub fn to_px(&self, units: f32) -> f32 {
        units * self.height_px * 0.5
    }

    // -- low level ----------------------------------------------------------

    /// Straight alpha -> premultiplied.
    fn pm(c: [f32; 4]) -> [f32; 4] {
        [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]]
    }

    /// Straight alpha -> premultiplied with zero destination alpha, which the
    /// premultiplied blend state composites additively.
    fn pm_add(c: [f32; 4]) -> [f32; 4] {
        [c[0] * c[3], c[1] * c[3], c[2] * c[3], 0.0]
    }

    /// Fully transparent in premultiplied space: contributes nothing under
    /// either blend mode, so the same value works for over and additive edges.
    const CLEAR: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

    fn push_vertex(&mut self, position: [f32; 2], premultiplied: [f32; 4]) -> u32 {
        let i = self.verts.len() as u32;
        self.verts.push(Vertex {
            position,
            color: premultiplied,
        });
        i
    }

    fn push_tri(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend_from_slice(&[a, b, c]);
    }

    fn push_quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.indices.extend_from_slice(&[a, b, c, a, c, d]);
    }

    // -- compatibility primitives -------------------------------------------

    pub fn tri(&mut self, a: [f32; 2], b: [f32; 2], c: [f32; 2], color: [f32; 4]) {
        let c4 = Self::pm(color);
        let (ia, ib, ic) = (
            self.push_vertex(a, c4),
            self.push_vertex(b, c4),
            self.push_vertex(c, c4),
        );
        self.push_tri(ia, ib, ic);
    }

    pub fn quad(&mut self, p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], color: [f32; 4]) {
        let c4 = Self::pm(color);
        let (a, b, c, d) = (
            self.push_vertex(p0, c4),
            self.push_vertex(p1, c4),
            self.push_vertex(p2, c4),
            self.push_vertex(p3, c4),
        );
        self.push_quad(a, b, c, d);
    }

    /// Hard-edged axis-aligned rectangle. Used by the HUD, which wants crisp
    /// pixel-snapped edges rather than a feathered ramp.
    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4]) {
        self.quad([x0, y1], [x1, y1], [x1, y0], [x0, y0], color);
    }

    /// Point on a circle of radius `r` (NDC-y units) around `center`,
    /// aspect-corrected so it stays round on a non-square window.
    pub fn polar(&self, center: [f32; 2], r: f32, angle: f32) -> [f32; 2] {
        [
            center[0] + r * angle.cos() / self.aspect,
            center[1] + r * angle.sin(),
        ]
    }

    // -- feathered convex fills ---------------------------------------------

    /// Fill a convex outline, with a one-pixel alpha ramp around its perimeter.
    ///
    /// `outline` is in NDC, counter-clockwise, and must be convex — the core is
    /// a triangle fan from the centroid. Outward normals are taken as the
    /// normalized sum of the two adjacent edge normals, which is exact for a
    /// convex polygon.
    pub fn fill_outline(&mut self, outline: &[[f32; 2]], color: [f32; 4], additive: bool) {
        let n = outline.len();
        if n < 3 {
            return;
        }
        let core = if additive {
            Self::pm_add(color)
        } else {
            Self::pm(color)
        };
        let feather = FEATHER_PX * self.px();

        let mut cx = 0.0;
        let mut cy = 0.0;
        for p in outline {
            cx += p[0];
            cy += p[1];
        }
        let centroid = [cx / n as f32, cy / n as f32];
        let center = self.push_vertex(centroid, core);

        let first_core = self.verts.len() as u32;
        for i in 0..n {
            let prev = outline[(i + n - 1) % n];
            let cur = outline[i];
            let next = outline[(i + 1) % n];

            // Edge normals in screen space (x scaled by aspect) so the fringe
            // is one pixel wide in both axes on a non-square window.
            let e0 = self.screen_normal(prev, cur);
            let e1 = self.screen_normal(cur, next);
            let mut nx = e0[0] + e1[0];
            let mut ny = e0[1] + e1[1];
            let len = (nx * nx + ny * ny).sqrt();
            if len < 1e-9 {
                nx = e1[0];
                ny = e1[1];
            } else {
                nx /= len;
                ny /= len;
            }

            self.push_vertex(cur, core);
            self.push_vertex(
                [
                    cur[0] + nx * feather / self.aspect,
                    cur[1] + ny * feather,
                ],
                Self::CLEAR,
            );
        }

        for i in 0..n {
            let a = first_core + (i as u32) * 2;
            let b = first_core + (((i + 1) % n) as u32) * 2;
            self.push_tri(center, a, b);
            self.push_quad(a, b, b + 1, a + 1);
        }
    }

    /// Outward unit normal of the edge `p0 -> p1`, in screen space.
    fn screen_normal(&self, p0: [f32; 2], p1: [f32; 2]) -> [f32; 2] {
        let dx = (p1[0] - p0[0]) * self.aspect;
        let dy = p1[1] - p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            return [0.0, 0.0];
        }
        // Right-hand normal of a counter-clockwise outline points outward.
        [dy / len, -dx / len]
    }

    /// Axis-aligned rectangle with rounded corners and feathered edges.
    ///
    /// `radius_px` is clamped to half the shorter side so a short bar collapses
    /// to a stadium shape instead of self-intersecting.
    pub fn round_rect(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        radius_px: f32,
        color: [f32; 4],
    ) {
        let (x0, x1) = (x0.min(x1), x0.max(x1));
        let (y0, y1) = (y0.min(y1), y0.max(y1));
        let w_px = self.to_px((x1 - x0) * self.aspect);
        let h_px = self.to_px(y1 - y0);
        if w_px <= 0.0 || h_px <= 0.0 {
            return;
        }
        let r_px = radius_px.min(w_px * 0.5).min(h_px * 0.5).max(0.0);
        let r = r_px * self.px();

        let slices = ((std::f32::consts::FRAC_PI_2 * r_px / CAP_SEGMENT_PX).ceil() as usize)
            .clamp(1, 8);

        let mut outline: Vec<[f32; 2]> = Vec::with_capacity(slices * 4 + 4);
        // Corner centers, counter-clockwise from bottom-right.
        let corners = [
            ([x1 - r / self.aspect, y0 + r], -std::f32::consts::FRAC_PI_2),
            ([x1 - r / self.aspect, y1 - r], 0.0),
            ([x0 + r / self.aspect, y1 - r], std::f32::consts::FRAC_PI_2),
            ([x0 + r / self.aspect, y0 + r], std::f32::consts::PI),
        ];
        for (center, start) in corners {
            for k in 0..=slices {
                let a = start + std::f32::consts::FRAC_PI_2 * k as f32 / slices as f32;
                outline.push(self.polar(center, r, a));
            }
        }
        self.fill_outline(&outline, color, false);
    }

    // -- strokes ------------------------------------------------------------

    /// Effective geometry width and alpha scale for a requested pixel width.
    ///
    /// See [`MIN_STROKE_PX`]: sub-pixel strokes are widened and dimmed by the
    /// same factor, holding emitted light constant.
    fn resolve_width(width_px: f32) -> (f32, f32) {
        let w = width_px.max(0.0);
        if w < MIN_STROKE_PX && w > 0.0 {
            (MIN_STROKE_PX, w / MIN_STROKE_PX)
        } else {
            (w.max(MIN_STROKE_PX), 1.0)
        }
    }

    /// Thick antialiased polyline with miter joins and round caps.
    ///
    /// Consecutive segments **share** their offset vertices at each joint, so
    /// there is no overlap and no double-blending — the old square-stamp joints
    /// beaded visibly wherever alpha was below 1. Where a miter would exceed
    /// [`MITER_LIMIT`] it is truncated instead.
    pub fn stroke(&mut self, pts: &[[f32; 2]], width_px: f32, color: [f32; 4], additive: bool) {
        self.stroke_with(pts, width_px, additive, |_| color);
    }

    /// As [`stroke`](Self::stroke), with color varying along the length.
    /// The closure receives 0.0 at the start and 1.0 at the end.
    pub fn stroke_with(
        &mut self,
        pts: &[[f32; 2]],
        width_px: f32,
        additive: bool,
        color_at: impl Fn(f32) -> [f32; 4],
    ) {
        if pts.len() < 2 || width_px <= 0.0 {
            return;
        }
        let (geo_px, alpha_scale) = Self::resolve_width(width_px);
        let half = geo_px * 0.5 * self.px();
        let feather = FEATHER_PX * self.px();
        let aspect = self.aspect;

        let encode = |c: [f32; 4]| {
            let c = [c[0], c[1], c[2], c[3] * alpha_scale];
            if additive {
                Self::pm_add(c)
            } else {
                Self::pm(c)
            }
        };

        // Segment directions in screen space, skipping degenerate spans.
        let mut normals = std::mem::take(&mut self.normals);
        normals.clear();
        for w in pts.windows(2) {
            let dx = (w[1][0] - w[0][0]) * aspect;
            let dy = w[1][1] - w[0][1];
            let len = (dx * dx + dy * dy).sqrt();
            normals.push(if len < 1e-9 {
                [0.0, 0.0]
            } else {
                [-dy / len, dx / len]
            });
        }
        // Carry the previous valid normal across zero-length spans so a
        // repeated point cannot collapse the ribbon.
        let mut last_good = [0.0f32, 1.0];
        for n in normals.iter_mut() {
            if n[0].abs() + n[1].abs() < 1e-9 {
                *n = last_good;
            } else {
                last_good = *n;
            }
        }

        let mut ribs = std::mem::take(&mut self.ribs);
        ribs.clear();
        let last = (pts.len() - 1) as f32;

        for (i, &p) in pts.iter().enumerate() {
            let (offset, scale) = if i == 0 {
                (normals[0], 1.0)
            } else if i == pts.len() - 1 {
                (normals[normals.len() - 1], 1.0)
            } else {
                let a = normals[i - 1];
                let b = normals[i];
                let mut mx = a[0] + b[0];
                let mut my = a[1] + b[1];
                let len = (mx * mx + my * my).sqrt();
                if len < 1e-6 {
                    // Exact reversal: fall back to the incoming normal.
                    (a, 1.0)
                } else {
                    mx /= len;
                    my /= len;
                    // Miter length = 1 / cos(theta/2) = 1 / dot(miter, normal).
                    let cos_half = (mx * a[0] + my * a[1]).abs().max(1e-4);
                    ((([mx, my])), (1.0 / cos_half).min(MITER_LIMIT))
                }
            };

            let c = encode(color_at(i as f32 / last));
            let at = |d: f32| {
                [
                    p[0] + offset[0] * d / aspect,
                    p[1] + offset[1] * d,
                ]
            };
            let inner = half * scale;
            let outer = (half + feather) * scale;
            let r = [
                self.push_vertex(at(-outer), Self::CLEAR),
                self.push_vertex(at(-inner), c),
                self.push_vertex(at(inner), c),
                self.push_vertex(at(outer), Self::CLEAR),
            ];
            ribs.push(r);
        }

        for w in 0..ribs.len() - 1 {
            let (a, b) = (ribs[w], ribs[w + 1]);
            for k in 0..3 {
                self.push_quad(a[k], a[k + 1], b[k + 1], b[k]);
            }
        }

        // Round caps: a half-disc at each end, feathered like everything else.
        let start_dir = [-normals[0][1], normals[0][0]];
        let end_n = normals[normals.len() - 1];
        let end_dir = [end_n[1], -end_n[0]];
        let c0 = encode(color_at(0.0));
        let c1 = encode(color_at(1.0));
        self.round_cap(pts[0], normals[0], start_dir, half, feather, c0, geo_px);
        self.round_cap(
            pts[pts.len() - 1],
            end_n,
            end_dir,
            half,
            feather,
            c1,
            geo_px,
        );

        self.ribs = ribs;
        self.normals = normals;
    }

    /// Half-disc cap centered on `p`, sweeping from `+normal` through
    /// `out_dir` to `-normal`. Both vectors are unit length in screen space.
    #[allow(clippy::too_many_arguments)]
    fn round_cap(
        &mut self,
        p: [f32; 2],
        normal: [f32; 2],
        out_dir: [f32; 2],
        half: f32,
        feather: f32,
        core: [f32; 4],
        geo_px: f32,
    ) {
        let slices = ((std::f32::consts::PI * geo_px * 0.5 / CAP_SEGMENT_PX).ceil() as usize)
            .clamp(2, 16);
        let center = self.push_vertex(p, core);
        let aspect = self.aspect;

        let first = self.verts.len() as u32;
        for k in 0..=slices {
            let a = std::f32::consts::PI * k as f32 / slices as f32;
            let (s, c) = a.sin_cos();
            let dir = [normal[0] * c + out_dir[0] * s, normal[1] * c + out_dir[1] * s];
            let at = |d: f32| [p[0] + dir[0] * d / aspect, p[1] + dir[1] * d];
            self.push_vertex(at(half), core);
            self.push_vertex(at(half + feather), Self::CLEAR);
        }
        for k in 0..slices {
            let a = first + (k as u32) * 2;
            let b = a + 2;
            self.push_tri(center, a, b);
            self.push_quad(a, b, b + 1, a + 1);
        }
    }

    /// Stacked strokes reading as a glowing neon line.
    ///
    /// The wide layers are emitted **additively** so they accumulate into a
    /// halo. Under ordinary "over" blending each layer would occlude the one
    /// beneath it and the result reads flat, which is what the previous
    /// stacked-polyline approach did.
    pub fn glow_stroke(&mut self, pts: &[[f32; 2]], width_px: f32, color: [f32; 4], layers: u32) {
        for layer in (1..layers.max(1) + 1).rev() {
            let spread = 1.0 + layer as f32 * 2.4;
            let fade = color[3] / (1.0 + layer as f32 * 3.0);
            self.stroke(
                pts,
                width_px * spread,
                [color[0], color[1], color[2], fade],
                true,
            );
        }
        self.stroke(pts, width_px, color, false);
    }

    /// Fill between a curve and a horizontal baseline as one connected strip.
    ///
    /// Emitting an independent quad per span made adjacent quads share an edge;
    /// at alpha below 1 those shared edges double-blended into visible vertical
    /// seams. Sharing vertices removes them by construction.
    ///
    /// Deliberately **not** feathered. Every caller either strokes the same
    /// curve directly over this fill's top edge or butts it against another
    /// fill, so a ramp would be invisible at best and a seam at worst — and at
    /// two vertices per point instead of three it is a third of the upload.
    pub fn fill_under(
        &mut self,
        pts: &[[f32; 2]],
        baseline: f32,
        color_at: impl Fn(f32) -> [f32; 4],
    ) {
        if pts.len() < 2 {
            return;
        }
        let last = (pts.len() - 1) as f32;
        let first = self.verts.len() as u32;

        for (i, &p) in pts.iter().enumerate() {
            let c = Self::pm(color_at(i as f32 / last));
            self.push_vertex(p, c);
            self.push_vertex([p[0], baseline], c);
        }
        for i in 0..pts.len() - 1 {
            let a = first + (i as u32) * 2;
            let b = a + 2;
            self.push_quad(a, b, b + 1, a + 1);
        }
    }

    /// Ring outline as a closed antialiased stroke.
    ///
    /// Segment count is derived from the radius in pixels so the polygon is
    /// always below the visible-facet threshold without over-tessellating a
    /// small ring.
    pub fn ring(&mut self, center: [f32; 2], r: f32, width_px: f32, color: [f32; 4]) {
        let r_px = self.to_px(r).max(1.0);
        // ~3 px per chord keeps the sagitta well under a pixel.
        let n = ((std::f32::consts::TAU * r_px / 3.0).ceil() as usize).clamp(12, 256);
        let mut pts: Vec<[f32; 2]> = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            pts.push(self.polar(center, r, a));
        }
        self.stroke(&pts, width_px, color, false);
    }
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}
