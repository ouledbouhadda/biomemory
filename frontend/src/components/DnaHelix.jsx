/**
 * Realistic animated 3D DNA double helix with golden/amber tones.
 * Canvas-based with proper depth sorting, tube-like strands, and glow.
 */
import { useEffect, useRef } from 'react';

export default function DnaHelix({ width = 140, height = 50 }) {
  const canvasRef = useRef(null);
  const frameRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    let animId;
    const speed = 0.03;
    const STEPS = 200;       // smooth curve resolution
    const NUM_RUNGS = 16;    // base pair connections
    const TWISTS = 2.5;      // number of full helix turns visible

    const draw = () => {
      frameRef.current += speed;
      const time = frameRef.current;
      ctx.clearRect(0, 0, width, height);

      const centerY = height / 2;
      const amplitude = height * 0.35;
      const margin = 8;
      const helixLen = width - margin * 2;

      // Generate all points for both strands
      const strand1 = [];
      const strand2 = [];
      for (let i = 0; i <= STEPS; i++) {
        const frac = i / STEPS;
        const x = margin + frac * helixLen;
        const angle = frac * Math.PI * 2 * TWISTS + time;
        const sin1 = Math.sin(angle);
        const cos1 = Math.cos(angle);
        const sin2 = Math.sin(angle + Math.PI);
        const cos2 = Math.cos(angle + Math.PI);

        strand1.push({ x, y: centerY + sin1 * amplitude, z: cos1 });
        strand2.push({ x, y: centerY + sin2 * amplitude, z: cos2 });
      }

      // Generate rungs (base pairs)
      const rungs = [];
      for (let i = 0; i < NUM_RUNGS; i++) {
        const frac = (i + 0.5) / NUM_RUNGS;
        const x = margin + frac * helixLen;
        const angle = frac * Math.PI * 2 * TWISTS + time;
        const sin1 = Math.sin(angle);
        const cos1 = Math.cos(angle);
        const y1 = centerY + sin1 * amplitude;
        const y2 = centerY - sin1 * amplitude;
        // Average z of the two connection points
        const z = (cos1 + Math.cos(angle + Math.PI)) / 2;
        rungs.push({ x, y1, y2, z, cos1 });
      }

      // === PASS 1: Draw everything BEHIND (z < 0) ===

      // Back strand 1 segments (tube effect)
      drawStrandTube(ctx, strand1, -1, '#a06820', '#c08030', amplitude);
      // Back strand 2 segments
      drawStrandTube(ctx, strand2, -1, '#a06820', '#c08030', amplitude);

      // Back rungs
      for (const rung of rungs) {
        if (rung.cos1 < 0.1 && rung.cos1 > -0.1) continue; // skip rungs at crossing
        const behindness = Math.max(0, -rung.cos1);
        if (behindness > 0) {
          ctx.beginPath();
          ctx.moveTo(rung.x, rung.y1);
          ctx.lineTo(rung.x, rung.y2);
          ctx.strokeStyle = `rgba(200, 170, 100, ${0.15 + behindness * 0.15})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // === PASS 2: Draw front rungs ===
      for (const rung of rungs) {
        const frontness = Math.max(0, rung.cos1);
        if (frontness > 0.1) {
          const alpha = 0.3 + frontness * 0.5;
          // Rung line
          ctx.beginPath();
          ctx.moveTo(rung.x, rung.y1);
          ctx.lineTo(rung.x, rung.y2);
          ctx.strokeStyle = `rgba(255, 220, 150, ${alpha})`;
          ctx.lineWidth = 1.2 + frontness * 0.8;
          ctx.stroke();

          // Glow dots at connection points
          const dotR = 1.5 + frontness * 2;
          const glowR = dotR * 2.5;
          for (const py of [rung.y1, rung.y2]) {
            // Outer glow
            const grd = ctx.createRadialGradient(rung.x, py, 0, rung.x, py, glowR);
            grd.addColorStop(0, `rgba(255, 230, 180, ${frontness * 0.6})`);
            grd.addColorStop(1, 'rgba(255, 230, 180, 0)');
            ctx.beginPath();
            ctx.arc(rung.x, py, glowR, 0, Math.PI * 2);
            ctx.fillStyle = grd;
            ctx.fill();
            // Bright core
            ctx.beginPath();
            ctx.arc(rung.x, py, dotR, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 245, 220, ${0.6 + frontness * 0.4})`;
            ctx.fill();
          }
        }
      }

      // === PASS 3: Draw everything IN FRONT (z >= 0) ===
      drawStrandTube(ctx, strand1, 1, '#d4952a', '#f5cc60', amplitude);
      drawStrandTube(ctx, strand2, 1, '#d4952a', '#f5cc60', amplitude);

      animId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animId);
  }, [width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height, display: 'block' }}
    />
  );
}

/**
 * Draw one strand as a thick tube with 3D shading.
 * side: -1 = only draw where z < 0 (back), 1 = only where z >= 0 (front)
 */
function drawStrandTube(ctx, points, side, colorDark, colorLight, amplitude) {
  // Draw the strand as a series of short thick segments with varying width/color
  for (let i = 1; i < points.length; i++) {
    const p0 = points[i - 1];
    const p1 = points[i];
    const avgZ = (p0.z + p1.z) / 2;

    // Filter by side
    if (side === 1 && avgZ < -0.05) continue;
    if (side === -1 && avgZ >= -0.05) continue;

    // Tube thickness based on depth: thicker when in front
    const depthFactor = (avgZ + 1) / 2; // 0..1
    const thickness = 1.5 + depthFactor * 4;

    // Color: brighter when in front
    const brightness = Math.max(0, Math.min(1, depthFactor));

    // Interpolate between dark and light color
    const r1 = parseInt(colorDark.slice(1, 3), 16);
    const g1 = parseInt(colorDark.slice(3, 5), 16);
    const b1 = parseInt(colorDark.slice(5, 7), 16);
    const r2 = parseInt(colorLight.slice(1, 3), 16);
    const g2 = parseInt(colorLight.slice(3, 5), 16);
    const b2 = parseInt(colorLight.slice(5, 7), 16);
    const r = Math.round(r1 + (r2 - r1) * brightness);
    const g = Math.round(g1 + (g2 - g1) * brightness);
    const b = Math.round(b1 + (b2 - b1) * brightness);

    const alpha = side === -1 ? 0.25 + brightness * 0.2 : 0.6 + brightness * 0.4;

    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    ctx.lineWidth = thickness;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Highlight shimmer on front-facing segments
    if (side === 1 && brightness > 0.7) {
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y - thickness * 0.2);
      ctx.lineTo(p1.x, p1.y - thickness * 0.2);
      ctx.strokeStyle = `rgba(255, 250, 230, ${(brightness - 0.7) * 1.5})`;
      ctx.lineWidth = thickness * 0.3;
      ctx.lineCap = 'round';
      ctx.stroke();
    }
  }
}
