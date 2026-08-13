//! Drawing primitives and the visualizer interface.
//!
//! Everything is emitted as plain triangles into one `MeshBuilder`, drawn in a
//! single draw call with painter's-algorithm ordering (no depth buffer): later
//! geometry paints over earlier geometry. That keeps the GPU side trivial and
//! means a new visual only has to know how to push triangles.

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

/// Per-frame audio and timing state handed to every visualizer.
///
/// Deliberately exposes more than any current visual consumes — the point is
/// that a new one can reach for stereo split or elapsed time without having to
/// thread new plumbing through `main`.
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
/// peak holds, rotation angles) — they're constructed once and reused.
pub trait Visualizer {
    fn name(&self) -> &'static str;
    fn draw(&mut self, frame: &FrameData, mesh: &mut MeshBuilder);
}

/// Accumulates triangles in normalized device coordinates (-1..1 on both axes).
///
/// Holds the viewport aspect ratio so that thickness and radius helpers stay
/// visually uniform instead of stretching with the window.
///
/// Some helpers here are unused by the shipped visuals; they exist so new ones
/// have primitives to build from.
#[allow(dead_code)]
pub struct MeshBuilder {
    verts: Vec<Vertex>,
    aspect: f32,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            verts: Vec::new(),
            aspect: 1.0,
        }
    }

    pub fn begin(&mut self, aspect: f32) {
        self.verts.clear();
        self.aspect = aspect.max(1e-3);
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.verts
    }

    #[allow(dead_code)]
    pub fn aspect(&self) -> f32 {
        self.aspect
    }

    pub fn tri(&mut self, a: [f32; 2], b: [f32; 2], c: [f32; 2], color: [f32; 4]) {
        self.verts.push(Vertex { position: a, color });
        self.verts.push(Vertex { position: b, color });
        self.verts.push(Vertex { position: c, color });
    }

    /// Quad from four corners given in order around the perimeter.
    pub fn quad(&mut self, p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2], color: [f32; 4]) {
        self.tri(p0, p1, p2, color);
        self.tri(p0, p2, p3, color);
    }

    /// Axis-aligned rectangle.
    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: [f32; 4]) {
        self.quad([x0, y1], [x1, y1], [x1, y0], [x0, y0], color);
    }

    /// Perpendicular offset for a segment, sized so the line looks equally
    /// thick regardless of window aspect.
    ///
    /// The direction is measured in screen space (x scaled by aspect), and the
    /// resulting normal converted back to NDC — otherwise a vertical and a
    /// horizontal line of the same nominal thickness render at different widths.
    fn normal(&self, p0: [f32; 2], p1: [f32; 2], half: f32) -> [f32; 2] {
        let dx = (p1[0] - p0[0]) * self.aspect;
        let dy = p1[1] - p0[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            return [0.0, 0.0];
        }
        [-dy / len * half / self.aspect, dx / len * half]
    }

    /// Single thick segment. `thickness` is in NDC-y units.
    pub fn line(&mut self, p0: [f32; 2], p1: [f32; 2], thickness: f32, color: [f32; 4]) {
        let n = self.normal(p0, p1, thickness * 0.5);
        self.quad(
            [p0[0] + n[0], p0[1] + n[1]],
            [p1[0] + n[0], p1[1] + n[1]],
            [p1[0] - n[0], p1[1] - n[1]],
            [p0[0] - n[0], p0[1] - n[1]],
            color,
        );
    }

    /// Small square centered on a point, used to fill polyline joints.
    fn joint(&mut self, p: [f32; 2], thickness: f32, color: [f32; 4]) {
        let hx = thickness * 0.5 / self.aspect;
        let hy = thickness * 0.5;
        self.rect(p[0] - hx, p[1] - hy, p[0] + hx, p[1] + hy, color);
    }

    /// Thick polyline with a constant color.
    pub fn polyline(&mut self, pts: &[[f32; 2]], thickness: f32, color: [f32; 4]) {
        self.polyline_with(pts, thickness, |_| color);
    }

    /// Thick polyline whose color varies along its length.
    ///
    /// The closure receives 0.0 at the start of the line and 1.0 at the end.
    pub fn polyline_with(
        &mut self,
        pts: &[[f32; 2]],
        thickness: f32,
        color_at: impl Fn(f32) -> [f32; 4],
    ) {
        if pts.len() < 2 {
            return;
        }
        let last = (pts.len() - 1) as f32;
        for i in 0..pts.len() - 1 {
            let color = color_at(i as f32 / last);
            self.line(pts[i], pts[i + 1], thickness, color);
            // Square off the joint so segments at an angle don't leave a notch.
            if i > 0 {
                self.joint(pts[i], thickness, color);
            }
        }
    }

    /// Polyline drawn as several stacked passes: wide and faint underneath,
    /// narrow and bright on top. Reads as a glowing neon stroke.
    pub fn glow_polyline(
        &mut self,
        pts: &[[f32; 2]],
        thickness: f32,
        color: [f32; 4],
        layers: u32,
    ) {
        for layer in (0..layers.max(1)).rev() {
            let spread = 1.0 + layer as f32 * 2.2;
            let fade = color[3] / (1.0 + layer as f32 * 2.6);
            let c = [color[0], color[1], color[2], fade];
            self.polyline(pts, thickness * spread, c);
        }
    }

    /// Fill the region between a polyline and a horizontal baseline.
    pub fn fill_under(&mut self, pts: &[[f32; 2]], baseline: f32, color: [f32; 4]) {
        for w in pts.windows(2) {
            self.quad(
                w[0],
                w[1],
                [w[1][0], baseline],
                [w[0][0], baseline],
                color,
            );
        }
    }

    /// Point on a circle of radius `r` around `center`, aspect-corrected so it
    /// stays round on a non-square window.
    pub fn polar(&self, center: [f32; 2], r: f32, angle: f32) -> [f32; 2] {
        [
            center[0] + r * angle.cos() / self.aspect,
            center[1] + r * angle.sin(),
        ]
    }

    /// Ring outline, as a closed thick polyline.
    pub fn ring(&mut self, center: [f32; 2], r: f32, thickness: f32, color: [f32; 4], segments: usize) {
        let n = segments.max(3);
        let pts: Vec<[f32; 2]> = (0..=n)
            .map(|i| {
                let a = i as f32 / n as f32 * std::f32::consts::TAU;
                self.polar(center, r, a)
            })
            .collect();
        self.polyline(&pts, thickness, color);
    }
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}
