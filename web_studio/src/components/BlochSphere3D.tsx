import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Download, Maximize2, Minimize2, RotateCcw } from 'lucide-react';

interface BlochSphere3DProps {
  qubitName: string;
  /** Full trajectory points [x, y, z] */
  trajectory: [number, number, number][];
  /** Index in trajectory corresponding to current timeline cursor (inclusive) */
  cursorIdx: number;
  /** Current state Bloch vector [x, y, z] */
  currentVector: [number, number, number];
  /** Optional step checkpoint indices along trajectory */
  stepIndices?: number[];
  darkMode: boolean;
  expanded?: boolean;
  onToggleExpand?: () => void;
}

type ViewPreset = '3d' | 'xy' | 'xz' | 'yz';

const VIEW_ANGLES: Record<ViewPreset, { azim: number; elev: number }> = {
  // Classic physics Bloch angle: +Z up, +X front-left-down, +Y front-right
  '3d': { azim: (38 * Math.PI) / 180, elev: (21 * Math.PI) / 180 },
  xy: { azim: (0 * Math.PI) / 180, elev: (88 * Math.PI) / 180 },
  xz: { azim: (-90 * Math.PI) / 180, elev: (4 * Math.PI) / 180 },
  yz: { azim: (0 * Math.PI) / 180, elev: (4 * Math.PI) / 180 },
};

/**
 * Project 3D Bloch point (x, y, z) into camera coordinates (sx, sy, depth)
 * where +Z is vertical up on screen (elev=0), +X points toward front-left at azim=38 deg,
 * +Y points toward front-right.
 */
function projectPoint(
  x: number,
  y: number,
  z: number,
  azim: number,
  elev: number
): { sx: number; sy: number; depth: number } {
  const ca = Math.cos(azim);
  const sa = Math.sin(azim);
  const ce = Math.cos(elev);
  const se = Math.sin(elev);

  // Camera right = (-sin a, cos a, 0) so at a=38deg, +X goes left-front, +Y goes right-front
  const sx = -sa * x + ca * y;
  // Camera forward (toward viewer) in XY plane = (cos a, sin a, 0)
  const fwdXY = ca * x + sa * y;
  // Screen up (positive upwards; canvas will negate)
  const sy = ce * z - se * fwdXY;
  // Depth toward viewer (positive = in front of sphere center)
  const depth = se * z + ce * fwdXY;

  // Mild perspective factor for natural 3D depth without distorting the unit sphere
  const persp = 1 / (1 - depth * 0.14);
  return { sx: sx * persp, sy: sy * persp, depth };
}

function lerpColor(c1: [number, number, number], c2: [number, number, number], t: number): string {
  const r = Math.round(c1[0] + (c2[0] - c1[0]) * t);
  const g = Math.round(c1[1] + (c2[1] - c1[1]) * t);
  const b = Math.round(c1[2] + (c2[2] - c1[2]) * t);
  return `rgb(${r},${g},${b})`;
}

export const BlochSphere3D: React.FC<BlochSphere3DProps> = ({
  qubitName,
  trajectory,
  cursorIdx,
  currentVector,
  stepIndices = [],
  darkMode,
  expanded = false,
  onToggleExpand,
}) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [viewPreset, setViewPreset] = useState<ViewPreset>('3d');
  const [camera, setCamera] = useState(VIEW_ANGLES['3d']);
  const cameraRef = useRef(VIEW_ANGLES['3d']);
  cameraRef.current = camera;

  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{
    x: number;
    y: number;
    azim: number;
    elev: number;
    lastX: number;
    lastY: number;
    lastT: number;
    vx: number;
    vy: number;
  } | null>(null);
  const camAnimRef = useRef<number | null>(null);
  const [size, setSize] = useState({ w: 320, h: 280 });

  // Smooth vector interpolation when switching observed qubit
  const [displayVector, setDisplayVector] = useState<[number, number, number]>(currentVector);
  const displayVecRef = useRef<[number, number, number]>(currentVector);
  displayVecRef.current = displayVector;
  const prevQubitRef = useRef<string>(qubitName);
  const vecAnimRef = useRef<number | null>(null);

  useEffect(() => {
    if (prevQubitRef.current !== qubitName) {
      prevQubitRef.current = qubitName;
      if (vecAnimRef.current) cancelAnimationFrame(vecAnimRef.current);
      const startVec: [number, number, number] = [...displayVecRef.current];
      const targetVec: [number, number, number] = [...currentVector];
      const t0 = performance.now();
      const dur = 220;
      const step = (now: number) => {
        const p = Math.min(1, (now - t0) / dur);
        const ease = 1 - Math.pow(1 - p, 3);
        setDisplayVector([
          startVec[0] + (targetVec[0] - startVec[0]) * ease,
          startVec[1] + (targetVec[1] - startVec[1]) * ease,
          startVec[2] + (targetVec[2] - startVec[2]) * ease,
        ]);
        if (p < 1) {
          vecAnimRef.current = requestAnimationFrame(step);
        } else {
          vecAnimRef.current = null;
        }
      };
      vecAnimRef.current = requestAnimationFrame(step);
    } else {
      if (vecAnimRef.current) {
        cancelAnimationFrame(vecAnimRef.current);
        vecAnimRef.current = null;
      }
      setDisplayVector(currentVector);
    }
  }, [qubitName, currentVector]);

  useEffect(() => {
    return () => {
      if (camAnimRef.current) cancelAnimationFrame(camAnimRef.current);
      if (vecAnimRef.current) cancelAnimationFrame(vecAnimRef.current);
    };
  }, []);

  // Track container dimensions crisply via clientWidth/clientHeight (canvas is absolute so zero layout loop)
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => {
      const cw = Math.floor(el.clientWidth);
      const ch = Math.floor(el.clientHeight);
      if (cw > 10 && ch > 10) {
        setSize((prev) => (prev.w === cw && prev.h === ch ? prev : { w: cw, h: ch }));
      }
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const applyPreset = (p: ViewPreset) => {
    setViewPreset(p);
    if (camAnimRef.current) cancelAnimationFrame(camAnimRef.current);
    const startAzim = cameraRef.current.azim;
    const startElev = cameraRef.current.elev;
    const target = VIEW_ANGLES[p];
    // Shortest angular distance for azimuth
    let dAzim = ((target.azim - startAzim + Math.PI) % (2 * Math.PI)) - Math.PI;
    if (dAzim < -Math.PI) dAzim += 2 * Math.PI;
    const dElev = target.elev - startElev;

    if (Math.hypot(dAzim, dElev) < 1e-3) {
      setCamera(target);
      return;
    }

    const t0 = performance.now();
    const dur = 320;
    const tick = (now: number) => {
      const prog = Math.min(1, (now - t0) / dur);
      const ease = 1 - Math.pow(1 - prog, 3);
      setCamera({
        azim: startAzim + dAzim * ease,
        elev: startElev + dElev * ease,
      });
      if (prog < 1) {
        camAnimRef.current = requestAnimationFrame(tick);
      } else {
        camAnimRef.current = null;
      }
    };
    camAnimRef.current = requestAnimationFrame(tick);
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (camAnimRef.current) {
      cancelAnimationFrame(camAnimRef.current);
      camAnimRef.current = null;
    }
    e.currentTarget.setPointerCapture(e.pointerId);
    setIsDragging(true);
    const now = performance.now();
    dragRef.current = {
      x: e.clientX,
      y: e.clientY,
      azim: camera.azim,
      elev: camera.elev,
      lastX: e.clientX,
      lastY: e.clientY,
      lastT: now,
      vx: 0,
      vy: 0,
    };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const now = performance.now();
    const dt = Math.max(1, now - dragRef.current.lastT);
    const instVx = (-(e.clientX - dragRef.current.lastX) * 0.011) / dt;
    const instVy = ((e.clientY - dragRef.current.lastY) * 0.011) / dt;
    dragRef.current.vx = dragRef.current.vx * 0.5 + instVx * 0.5;
    dragRef.current.vy = dragRef.current.vy * 0.5 + instVy * 0.5;
    dragRef.current.lastX = e.clientX;
    dragRef.current.lastY = e.clientY;
    dragRef.current.lastT = now;

    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    const nextAzim = dragRef.current.azim - dx * 0.011;
    const nextElev = Math.max(
      -Math.PI / 2 + 0.04,
      Math.min(Math.PI / 2 - 0.04, dragRef.current.elev + dy * 0.011)
    );
    setCamera({ azim: nextAzim, elev: nextElev });
  };

  const handlePointerUp = () => {
    setIsDragging(false);
    const d = dragRef.current;
    dragRef.current = null;
    if (!d) return;
    let vx = d.vx;
    let vy = d.vy;
    if (Math.hypot(vx, vy) < 0.00018) return;

    let lastT = performance.now();
    const coast = (now: number) => {
      const dt = Math.min(32, Math.max(1, now - lastT));
      lastT = now;
      const decay = Math.pow(0.89, dt / 16);
      vx *= decay;
      vy *= decay;
      if (Math.hypot(vx, vy) < 0.00004) {
        camAnimRef.current = null;
        return;
      }
      setCamera((prev) => ({
        azim: prev.azim + vx * dt,
        elev: Math.max(-Math.PI / 2 + 0.04, Math.min(Math.PI / 2 - 0.04, prev.elev + vy * dt)),
      }));
      camAnimRef.current = requestAnimationFrame(coast);
    };
    camAnimRef.current = requestAnimationFrame(coast);
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const { w, h } = size;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2 + 2;
    // Generous sphere radius filling ~74% of the smaller dimension with room for labels
    const R = Math.max(52, Math.min(w, h) * 0.375);

    const toScreen = (x: number, y: number, z: number) => {
      const p = projectPoint(x, y, z, camera.azim, camera.elev);
      return { x: cx + p.sx * R, y: cy - p.sy * R, depth: p.depth };
    };

    // ------------------------------------------------------------------
    // 1. Soft Ambient Ground Shadow beneath Sphere
    // ------------------------------------------------------------------
    const shadowY = cy + R * 1.12;
    const shadowGrad = ctx.createRadialGradient(cx, shadowY, R * 0.05, cx, shadowY, R * 0.78);
    shadowGrad.addColorStop(0, darkMode ? 'rgba(0,0,0,0.38)' : 'rgba(15,23,42,0.11)');
    shadowGrad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.save();
    ctx.translate(cx, shadowY);
    ctx.scale(1, 0.24);
    ctx.beginPath();
    ctx.arc(0, 0, R * 0.78, 0, Math.PI * 2);
    ctx.fillStyle = shadowGrad;
    ctx.fill();
    ctx.restore();

    // ------------------------------------------------------------------
    // 2. Translucent 3D Glass Sphere Base Shading
    // ------------------------------------------------------------------
    const globeGrad = ctx.createRadialGradient(
      cx - R * 0.34,
      cy - R * 0.36,
      R * 0.06,
      cx,
      cy,
      R * 1.05
    );
    if (darkMode) {
      globeGrad.addColorStop(0, 'rgba(148, 163, 184, 0.20)');
      globeGrad.addColorStop(0.55, 'rgba(51, 65, 85, 0.18)');
      globeGrad.addColorStop(1, 'rgba(15, 23, 42, 0.38)');
    } else {
      globeGrad.addColorStop(0, 'rgba(255, 255, 255, 0.92)');
      globeGrad.addColorStop(0.45, 'rgba(226, 238, 252, 0.42)');
      globeGrad.addColorStop(0.85, 'rgba(203, 213, 225, 0.36)');
      globeGrad.addColorStop(1, 'rgba(148, 163, 184, 0.42)');
    }
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.fillStyle = globeGrad;
    ctx.fill();

    // Helper: draw 3D circle split into back-hemisphere (dashed) & front-hemisphere (solid)
    const drawCircle3D = (
      fn: (t: number) => [number, number, number],
      frontOnly: boolean,
      strokeColor: string,
      lineWidth: number,
      dashed = false,
      segments = 96
    ) => {
      ctx.save();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.setLineDash(dashed ? [3.5, 3.5] : []);
      let drawing = false;
      ctx.beginPath();
      for (let i = 0; i <= segments; i++) {
        const t = (2 * Math.PI * i) / segments;
        const [x, y, z] = fn(t);
        const p = toScreen(x, y, z);
        const visible = frontOnly ? p.depth >= -0.02 : p.depth < 0.02;
        if (visible) {
          if (!drawing) {
            ctx.moveTo(p.x, p.y);
            drawing = true;
          } else {
            ctx.lineTo(p.x, p.y);
          }
        } else if (drawing) {
          ctx.stroke();
          ctx.beginPath();
          drawing = false;
        }
      }
      if (drawing) ctx.stroke();
      ctx.restore();
    };

    const backWire = darkMode ? 'rgba(148,163,184,0.22)' : 'rgba(100,116,139,0.28)';
    const backLat = darkMode ? 'rgba(148,163,184,0.13)' : 'rgba(148,163,184,0.24)';
    const frontEquator = darkMode ? 'rgba(203,213,225,0.62)' : 'rgba(71,85,105,0.58)';
    const frontMeridian = darkMode ? 'rgba(148,163,184,0.36)' : 'rgba(100,116,139,0.36)';
    const frontLat = darkMode ? 'rgba(148,163,184,0.20)' : 'rgba(148,163,184,0.28)';

    // ------------------------------------------------------------------
    // 3. Back Hemisphere Wireframe (Equator, Meridians, ±45° Latitudes)
    // ------------------------------------------------------------------
    drawCircle3D((t) => [Math.cos(t), Math.sin(t), 0], false, backWire, 1.3, true);
    drawCircle3D((t) => [Math.cos(t), 0, Math.sin(t)], false, backWire, 1.0, true);
    drawCircle3D((t) => [0, Math.cos(t), Math.sin(t)], false, backWire, 1.0, true);
    const lat45 = Math.SQRT1_2;
    drawCircle3D((t) => [lat45 * Math.cos(t), lat45 * Math.sin(t), lat45], false, backLat, 0.9, true);
    drawCircle3D((t) => [lat45 * Math.cos(t), lat45 * Math.sin(t), -lat45], false, backLat, 0.9, true);

    // ------------------------------------------------------------------
    // 4. Translucent Equatorial Disc Tint (makes XY plane unmistakable)
    // ------------------------------------------------------------------
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i <= 72; i++) {
      const t = (2 * Math.PI * i) / 72;
      const p = toScreen(Math.cos(t), Math.sin(t), 0);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    ctx.fillStyle = darkMode ? 'rgba(56, 189, 248, 0.055)' : 'rgba(10, 132, 255, 0.055)';
    ctx.fill();
    ctx.restore();

    // ------------------------------------------------------------------
    // 5. Internal Coordinate Axes (-1..0 dashed, 0..+1 crisp)
    // ------------------------------------------------------------------
    const origin = toScreen(0, 0, 0);
    const axes: { neg: [number, number, number]; pos: [number, number, number]; color: string }[] = [
      { neg: [-1, 0, 0], pos: [1, 0, 0], color: darkMode ? 'rgba(96,165,250,0.55)' : 'rgba(59,130,246,0.50)' },
      { neg: [0, -1, 0], pos: [0, 1, 0], color: darkMode ? 'rgba(251,191,36,0.55)' : 'rgba(217,119,6,0.50)' },
      { neg: [0, 0, -1], pos: [0, 0, 1], color: darkMode ? 'rgba(52,211,153,0.55)' : 'rgba(16,185,129,0.52)' },
    ];
    for (const ax of axes) {
      const pNeg = toScreen(...ax.neg);
      const pPos = toScreen(...ax.pos);
      // Negative half dashed
      ctx.save();
      ctx.strokeStyle = darkMode ? 'rgba(148,163,184,0.28)' : 'rgba(100,116,139,0.30)';
      ctx.lineWidth = 1.05;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(pNeg.x, pNeg.y);
      ctx.lineTo(origin.x, origin.y);
      ctx.stroke();
      ctx.restore();

      // Positive half solid
      ctx.save();
      ctx.strokeStyle = ax.color;
      ctx.lineWidth = 1.45;
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(pPos.x, pPos.y);
      ctx.stroke();
      ctx.restore();
    }

    // Center origin dot
    ctx.beginPath();
    ctx.arc(origin.x, origin.y, 2.2, 0, Math.PI * 2);
    ctx.fillStyle = darkMode ? 'rgba(203,213,225,0.6)' : 'rgba(71,85,105,0.55)';
    ctx.fill();

    // ------------------------------------------------------------------
    // 6. Equatorial Projection Shadow of Trajectory & Current Vector
    // ------------------------------------------------------------------
    const [bx, by, bz] = displayVector;
    const vecLen = Math.hypot(bx, by, bz);
    const tip = toScreen(bx, by, bz);
    const eqProj = toScreen(bx, by, 0);

    const floatEnd = Math.max(0, Math.min(cursorIdx, trajectory.length - 1));
    const activeEnd = Math.floor(floatEnd);
    const hasSubSeg = floatEnd - activeEnd > 1e-3;

    if (trajectory.length > 1) {
      ctx.save();
      ctx.strokeStyle = darkMode ? 'rgba(148,163,184,0.22)' : 'rgba(100,116,139,0.24)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2.5, 2.5]);
      ctx.beginPath();
      for (let i = 0; i <= activeEnd; i++) {
        const [x, y] = trajectory[i];
        const sp = toScreen(x, y, 0);
        if (i === 0) ctx.moveTo(sp.x, sp.y);
        else ctx.lineTo(sp.x, sp.y);
      }
      if (hasSubSeg) {
        ctx.lineTo(eqProj.x, eqProj.y);
      }
      ctx.stroke();
      ctx.restore();
    }

    if (vecLen >= 0.04) {
      // Vertical drop line (tip -> equator) & radial equator line (origin -> equator)
      ctx.save();
      ctx.strokeStyle = darkMode ? 'rgba(248,113,113,0.48)' : 'rgba(220,38,38,0.38)';
      ctx.lineWidth = 1.15;
      ctx.setLineDash([3, 2.5]);
      ctx.beginPath();
      ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(eqProj.x, eqProj.y);
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(eqProj.x, eqProj.y);
      ctx.stroke();
      ctx.restore();

      // Equatorial shadow dot
      ctx.beginPath();
      ctx.arc(eqProj.x, eqProj.y, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = darkMode ? 'rgba(248,113,113,0.65)' : 'rgba(220,38,38,0.52)';
      ctx.fill();
    }

    // ------------------------------------------------------------------
    // 7. Evolution Trajectory (Progressive Real-Time Emergence up to Cursor)
    // ------------------------------------------------------------------
    if (trajectory.length > 1) {
      // Active elapsed trajectory up to cursorIdx (plus fractional tip segment)
      if (floatEnd >= 1e-3) {
        const cStart: [number, number, number] = [99, 102, 241]; // Indigo
        const cMid: [number, number, number] = [10, 132, 255]; // Apple Blue
        const cEnd: [number, number, number] = [6, 182, 212]; // Vivid Cyan

        // Soft outer glow under curve
        ctx.save();
        ctx.strokeStyle = darkMode ? 'rgba(10,132,255,0.28)' : 'rgba(10,132,255,0.18)';
        ctx.lineWidth = 6.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i <= activeEnd; i++) {
          const p = toScreen(...trajectory[i]);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        if (hasSubSeg) {
          ctx.lineTo(tip.x, tip.y);
        }
        ctx.stroke();
        ctx.restore();

        // Crisp segmented gradient stroke
        ctx.save();
        ctx.lineWidth = 3.1;
        ctx.lineCap = 'round';
        const denom = Math.max(1e-3, floatEnd);
        for (let i = 0; i < activeEnd; i++) {
          const t = i / denom;
          const segColor =
            t < 0.5
              ? lerpColor(cStart, cMid, t * 2)
              : lerpColor(cMid, cEnd, (t - 0.5) * 2);
          const p0 = toScreen(...trajectory[i]);
          const p1 = toScreen(...trajectory[i + 1]);
          ctx.strokeStyle = segColor;
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();
        }
        if (hasSubSeg) {
          const pLast = toScreen(...trajectory[activeEnd]);
          ctx.strokeStyle = lerpColor(cMid, cEnd, 1);
          ctx.beginPath();
          ctx.moveTo(pLast.x, pLast.y);
          ctx.lineTo(tip.x, tip.y);
          ctx.stroke();
        }
        ctx.restore();
      }

      // Start marker (t = 0)
      const pStart = toScreen(...trajectory[0]);
      ctx.beginPath();
      ctx.arc(pStart.x, pStart.y, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = darkMode ? '#1C1C1E' : '#FFFFFF';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#6366F1';
      ctx.stroke();

      // Step checkpoint dots along elapsed trajectory
      for (const sIdx of stepIndices) {
        if (sIdx > 0 && sIdx <= floatEnd && sIdx < trajectory.length) {
          const ps = toScreen(...trajectory[sIdx]);
          ctx.beginPath();
          ctx.arc(ps.x, ps.y, 2.7, 0, Math.PI * 2);
          ctx.fillStyle = darkMode ? '#E2E8F0' : '#FFFFFF';
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = '#0A84FF';
          ctx.stroke();
        }
      }
    }

    // ------------------------------------------------------------------
    // 8. Current Bloch State Vector Arrow (or Mixed-State Origin Marker)
    // ------------------------------------------------------------------
    if (vecLen >= 0.05) {
      const dx = tip.x - origin.x;
      const dy = tip.y - origin.y;
      const screenLen = Math.hypot(dx, dy);
      const ux = screenLen > 1e-3 ? dx / screenLen : 0;
      const uy = screenLen > 1e-3 ? dy / screenLen : -1;
      const headLen = Math.min(13, Math.max(7, screenLen * 0.25));
      const headW = headLen * 0.48;

      // Soft shadow shaft
      ctx.save();
      ctx.strokeStyle = 'rgba(220, 38, 38, 0.24)';
      ctx.lineWidth = 6.5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(tip.x - ux * (headLen * 0.6), tip.y - uy * (headLen * 0.6));
      ctx.stroke();

      // Crisp crimson shaft
      ctx.strokeStyle = '#EF4444';
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.lineTo(tip.x - ux * (headLen * 0.6), tip.y - uy * (headLen * 0.6));
      ctx.stroke();

      // 3D Arrowhead Cone
      const baseX = tip.x - ux * headLen;
      const baseY = tip.y - uy * headLen;
      ctx.fillStyle = '#DC2626';
      ctx.beginPath();
      ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(baseX - uy * headW, baseY + ux * headW);
      ctx.lineTo(baseX + uy * headW, baseY - ux * headW);
      ctx.closePath();
      ctx.fill();

      // Glowing Tip Dot
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 3.8, 0, Math.PI * 2);
      ctx.fillStyle = '#EF4444';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#FFFFFF';
      ctx.stroke();
      ctx.restore();
    } else {
      // Maximally mixed / entangled reduced state at center (|r| ≈ 0)
      ctx.save();
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 11, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.16)';
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.55)';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 5.2, 0, Math.PI * 2);
      ctx.fillStyle = '#EF4444';
      ctx.fill();
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = '#FFFFFF';
      ctx.stroke();
      ctx.restore();
    }

    // ------------------------------------------------------------------
    // 9. Front Hemisphere Wireframe + Glass Rim + Specular Highlight
    // ------------------------------------------------------------------
    drawCircle3D((t) => [Math.cos(t), Math.sin(t), 0], true, frontEquator, 1.75, false);
    drawCircle3D((t) => [Math.cos(t), 0, Math.sin(t)], true, frontMeridian, 1.15, false);
    drawCircle3D((t) => [0, Math.cos(t), Math.sin(t)], true, frontMeridian, 1.15, false);
    drawCircle3D((t) => [lat45 * Math.cos(t), lat45 * Math.sin(t), lat45], true, frontLat, 0.9, false);
    drawCircle3D((t) => [lat45 * Math.cos(t), lat45 * Math.sin(t), -lat45], true, frontLat, 0.9, false);

    // Outer glass silhouette rim
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.lineWidth = 1.45;
    ctx.strokeStyle = darkMode ? 'rgba(203,213,225,0.42)' : 'rgba(100,116,139,0.48)';
    ctx.stroke();

    // Subtle glass specular arc on top-left rim
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R - 3, Math.PI * 1.12, Math.PI * 1.38);
    ctx.strokeStyle = darkMode ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 2.4;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.restore();

    // ------------------------------------------------------------------
    // 10. Crisp Pole & Cardinal Basis Labels (depth-aware opacity)
    // ------------------------------------------------------------------
    const labels: { pos: [number, number, number]; text: string; sub?: string; primary?: boolean; color?: string }[] = [
      { pos: [0, 0, 1.17], text: '|0⟩', primary: true, color: darkMode ? '#34D399' : '#059669' },
      { pos: [0, 0, -1.18], text: '|1⟩', primary: true, color: darkMode ? '#34D399' : '#059669' },
      { pos: [1.18, 0, 0], text: '|+⟩', sub: '+X', primary: true, color: darkMode ? '#60A5FA' : '#2563EB' },
      { pos: [-1.16, 0, 0], text: '|-⟩', primary: false },
      { pos: [0, 1.18, 0], text: '|+i⟩', sub: '+Y', primary: true, color: darkMode ? '#FBBF24' : '#D97706' },
      { pos: [0, -1.16, 0], text: '|-i⟩', primary: false },
    ];

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const lb of labels) {
      const p = toScreen(...lb.pos);
      const distFromCenter = Math.hypot(p.x - cx, p.y - cy);
      // Skip labels projected straight onto the center (e.g. |0>/|1> in top-down XY view, or |+i>/|-i> in XZ view)
      if (distFromCenter < R * 0.28) continue;
      // Fade secondary labels when directly behind the sphere center
      if (!lb.primary && p.depth < -0.35 && distFromCenter < R * 0.92) continue;
      const alpha = p.depth < -0.3 ? 0.45 : lb.primary ? 1 : 0.72;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.font = lb.primary
        ? '600 12px "SF Mono", "JetBrains Mono", Consolas, monospace'
        : '500 10.5px "SF Mono", "JetBrains Mono", Consolas, monospace';

      // Soft pill halo behind text so it stays crystal-clear
      const displayStr = lb.sub ? `${lb.text}` : lb.text;
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = darkMode ? 'rgba(37,37,40,0.9)' : 'rgba(255,255,255,0.92)';
      ctx.strokeText(displayStr, p.x, p.y);
      ctx.fillStyle = lb.color || (darkMode ? '#E5E5EA' : '#1D1D1F');
      ctx.fillText(displayStr, p.x, p.y);
      ctx.restore();
    }

    ctx.restore();
  }, [size, camera, trajectory, cursorIdx, displayVector, stepIndices, darkMode]);

  useEffect(() => {
    draw();
  }, [draw]);

  const [bx, by, bz] = displayVector;
  const radius = Math.hypot(bx, by, bz);

  const handleExportPng = () => {
    const src = canvasRef.current;
    if (!src) return;
    const out = document.createElement('canvas');
    const pad = 28;
    const footerH = 44;
    out.width = src.width + pad * 2;
    out.height = src.height + pad * 2 + footerH;
    const ctx = out.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = darkMode ? '#252528' : '#FFFFFF';
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(src, pad, pad);
    const scale = window.devicePixelRatio || 1;
    ctx.fillStyle = darkMode ? '#E5E5EA' : '#1D1D1F';
    ctx.font = `600 ${Math.round(13 * scale)}px "SF Mono", Consolas, monospace`;
    ctx.textAlign = 'center';
    ctx.fillText(
      `Bloch ${qubitName}   X=${bx >= 0 ? '+' : ''}${bx.toFixed(3)}   Y=${by >= 0 ? '+' : ''}${by.toFixed(3)}   Z=${bz >= 0 ? '+' : ''}${bz.toFixed(3)}   |r|=${radius.toFixed(3)}`,
      out.width / 2,
      out.height - Math.round(16 * scale)
    );
    const link = document.createElement('a');
    link.download = `bloch_${qubitName}.png`;
    link.href = out.toDataURL('image/png');
    link.click();
  };

  return (
    <div className="h-full flex flex-col studio-card overflow-hidden">
      {/* Crisp macOS Card Header (Never truncates!) */}
      <div className="h-9 px-3.5 border-b border-[var(--border-hairline)] flex items-center justify-between gap-2 shrink-0 bg-[var(--bg-panel)]">
        <span className="text-[12.5px] font-semibold text-[var(--text-primary)] whitespace-nowrap">
          Bloch 球 · {qubitName}
        </span>

        <div className="flex items-center gap-1 shrink-0">
          <div className="bg-black/[0.05] dark:bg-white/[0.08] p-0.5 rounded-lg inline-flex">
            {(['3d', 'xy', 'xz', 'yz'] as ViewPreset[]).map((vp) => (
              <button
                key={vp}
                onClick={() => applyPreset(vp)}
                className={`px-1.5 py-0.5 rounded-md text-[10px] font-mono-num uppercase transition-colors cursor-pointer ${
                  viewPreset === vp
                    ? 'bg-[var(--bg-panel)] text-[var(--text-primary)] font-semibold shadow-2xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                {vp}
              </button>
            ))}
          </div>
          <button
            onClick={() => applyPreset('3d')}
            title="重置 3D 视角（也可拖拽旋转）"
            className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleExportPng}
            title="导出高清 Bloch 球 PNG 图片"
            className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[#0A84FF] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
          {onToggleExpand && (
            <button
              onClick={onToggleExpand}
              title={expanded ? '还原分屏视图' : '放大查看 Bloch 球'}
              className="p-1 rounded-md text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
            >
              {expanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          )}
        </div>
      </div>

      {/* Main High-DPI 3D Canvas Stage (Absolute 1:1 pixel ratio — zero oval stretching!) */}
      <div ref={wrapRef} className="relative flex-1 min-h-[210px] select-none overflow-hidden">
        <canvas
          ref={canvasRef}
          style={{ width: size.w, height: size.h }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onDoubleClick={() => applyPreset('3d')}
          className={`absolute inset-0 block ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
          title="按住左键拖拽可自由旋转 3D 视角，双击复位"
        />
        {radius < 0.06 ? (
          <div className="pointer-events-none absolute bottom-1.5 left-2.5 px-2 py-0.5 rounded-md bg-rose-500/12 border border-rose-500/25 text-rose-600 dark:text-rose-300 text-[10px] font-medium">
            纠缠 / 混态 (|r| ≈ 0)
          </div>
        ) : (
          <div className="pointer-events-none absolute bottom-1.5 right-2.5 text-[10px] text-[var(--text-secondary)]/75">
            拖拽旋转 · 双击复位
          </div>
        )}
      </div>

      {/* Live Telemetry Footer Strip */}
      <div className="px-3 py-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)]/70 flex items-center justify-between gap-1.5 text-[11px] font-mono-num shrink-0 overflow-hidden">
        <div className="flex items-center gap-2.5">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[#0A84FF]" />
            <span className="text-[var(--text-secondary)]">X</span>
            <span className="font-semibold text-[var(--text-primary)]">
              {bx >= 0 ? '+' : ''}
              {bx.toFixed(2)}
            </span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[#F59E0B]" />
            <span className="text-[var(--text-secondary)]">Y</span>
            <span className="font-semibold text-[var(--text-primary)]">
              {by >= 0 ? '+' : ''}
              {by.toFixed(2)}
            </span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[#10B981]" />
            <span className="text-[var(--text-secondary)]">Z</span>
            <span className="font-semibold text-[var(--text-primary)]">
              {bz >= 0 ? '+' : ''}
              {bz.toFixed(2)}
            </span>
          </span>
        </div>
        <span
          className="px-1.5 py-0.5 rounded bg-[var(--bg-panel)] border border-[var(--border-hairline)] text-[10.5px] text-[var(--text-secondary)]"
          title="Bloch 矢量模长 |r|（纯态为 1.00，最大纠缠/完全退相干混态趋近 0）"
        >
          |r|={radius.toFixed(2)}
        </span>
      </div>
    </div>
  );
};
