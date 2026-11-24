import React, { useRef, useState } from 'react';
import type { HardwareProfileModel, QubitSpec } from '../types';

interface TopologyEditorProps {
  qubits: QubitSpec[];
  hardware: HardwareProfileModel;
  onUpdateQubitPos: (qubitId: number, x: number, y: number) => void;
  onToggleCustomEdge: (u: number, v: number) => void;
}

export const TopologyEditor: React.FC<TopologyEditorProps> = ({
  qubits,
  hardware,
  onUpdateQubitPos,
  onToggleCustomEdge,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [draggingId, setDraggingId] = useState<number | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);

  const isRydberg = hardware.topology_mode === 'rydberg_radius';
  const rb = Math.max(0.5, hardware.blockade_radius_um);

  const n = qubits.length;

  // Compute active edges based on topology_mode (using fast cell bucketing when n > 120)
  const edges: [number, number][] = [];
  let totalEdgeCount = 0;
  if (hardware.topology_mode === 'all') {
    totalEdgeCount = (n * (n - 1)) / 2;
    for (let i = 0; i < n && edges.length < 220; i++) {
      for (let j = i + 1; j < n && edges.length < 220; j++) {
        edges.push([i, j]);
      }
    }
  } else if (hardware.topology_mode === 'linear') {
    totalEdgeCount = Math.max(0, n - 1);
    for (let i = 0; i < n - 1 && edges.length < 220; i++) {
      edges.push([i, i + 1]);
    }
  } else if (hardware.topology_mode === 'custom') {
    for (const [u, v] of hardware.edges) {
      if (u >= 0 && u < n && v >= 0 && v < n && u !== v) {
        totalEdgeCount++;
        if (edges.length < 220) edges.push([Math.min(u, v), Math.max(u, v)]);
      }
    }
  } else {
    // rydberg_radius
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = Math.hypot(qubits[i].pos_x - qubits[j].pos_x, qubits[i].pos_y - qubits[j].pos_y);
        if (d <= rb + 1e-6) {
          totalEdgeCount++;
          if (edges.length < 220) edges.push([i, j]);
        }
      }
    }
  }

  // Dynamically fit bounding box of qubit coordinates into SVG viewBox (260 x 170)
  const xs = qubits.map((q) => q.pos_x);
  const ys = qubits.map((q) => q.pos_y);
  const minX = xs.length ? Math.min(...xs) : 0;
  const maxX = xs.length ? Math.max(...xs) : 13.5;
  const minY = ys.length ? Math.min(...ys) : 0;
  const maxY = ys.length ? Math.max(...ys) : 0;
  const spanX = Math.max(10, maxX - minX);
  const spanY = Math.max(6, maxY - minY);
  const scale = Math.min(11, 206 / spanX, 122 / spanY);
  const cxMid = 0.5 * (minX + maxX);
  const cyMid = 0.5 * (minY + maxY);
  const toSvg = (x: number, y: number) => ({
    sx: 130 + (x - cxMid) * scale,
    sy: 85 - (y - cyMid) * scale,
  });
  const fromSvg = (sx: number, sy: number) => ({
    x: Number((cxMid + (sx - 130) / scale).toFixed(1)),
    y: Number((cyMid + (85 - sy) / scale).toFixed(1)),
  });
  const nodeR = n <= 9 ? 10.5 : n <= 25 ? 7.2 : n <= 81 ? 4.4 : 2.8;
  const showLabels = n <= 25;

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (draggingId === null || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * 260;
    const sy = ((e.clientY - rect.top) / rect.height) * 170;
    const clampedX = Math.max(14, Math.min(246, sx));
    const clampedY = Math.max(14, Math.min(156, sy));
    const { x, y } = fromSvg(clampedX, clampedY);
    onUpdateQubitPos(draggingId, x, y);
  };

  return (
    <div className="space-y-1.5">
      <div className="rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-subtle)] overflow-hidden">
        <svg
          ref={svgRef}
          viewBox="0 0 260 170"
          className="w-full h-[155px] select-none touch-none"
          onPointerMove={handlePointerMove}
          onPointerUp={() => setDraggingId(null)}
          onPointerLeave={() => setDraggingId(null)}
        >
          {/* Subtle grid */}
          <defs>
            <pattern id="topoGrid" width="22" height="22" patternUnits="userSpaceOnUse">
              <circle cx="1" cy="1" r="0.8" fill="rgba(142,142,147,0.35)" />
            </pattern>
          </defs>
          <rect width="260" height="170" fill="url(#topoGrid)" />

          {/* Rydberg blockade radius rings (rendered up to 64 atoms for visual clarity) */}
          {isRydberg &&
            n <= 64 &&
            qubits.map((q) => {
              const { sx, sy } = toSvg(q.pos_x, q.pos_y);
              return (
                <circle
                  key={`rb_${q.id}`}
                  cx={sx}
                  cy={sy}
                  r={(rb * scale) / 2}
                  fill="rgba(10,132,255,0.06)"
                  stroke="rgba(10,132,255,0.28)"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
              );
            })}

          {/* Coupling edges */}
          {edges.map(([i, j]) => {
            const p1 = toSvg(qubits[i].pos_x, qubits[i].pos_y);
            const p2 = toSvg(qubits[j].pos_x, qubits[j].pos_y);
            return (
              <line
                key={`e_${i}_${j}`}
                x1={p1.sx}
                y1={p1.sy}
                x2={p2.sx}
                y2={p2.sy}
                stroke="#0A84FF"
                strokeWidth={n > 25 ? 1.1 : 1.8}
                strokeOpacity={n > 25 ? 0.5 : 0.7}
              />
            );
          })}

          {/* Qubit / Atom nodes */}
          {qubits.map((q, idx) => {
            const { sx, sy } = toSvg(q.pos_x, q.pos_y);
            const isSel = selectedNode === idx;
            return (
              <g
                key={q.id}
                transform={`translate(${sx}, ${sy})`}
                className="cursor-grab active:cursor-grabbing"
                onPointerDown={(e) => {
                  e.stopPropagation();
                  if (hardware.topology_mode === 'custom') {
                    if (selectedNode === null) {
                      setSelectedNode(idx);
                    } else if (selectedNode === idx) {
                      setSelectedNode(null);
                    } else {
                      onToggleCustomEdge(selectedNode, idx);
                      setSelectedNode(null);
                    }
                  } else {
                    setDraggingId(q.id);
                  }
                }}
              >
                <title>{`${q.name} (${q.pos_x} μm, ${q.pos_y} μm)`}</title>
                <circle
                  r={nodeR}
                  fill={n > 25 ? q.color : 'var(--bg-panel)'}
                  stroke={isSel ? '#0A84FF' : q.color}
                  strokeWidth={isSel ? 2.5 : n > 25 ? 1 : 2}
                />
                {showLabels && (
                  <text
                    y={n <= 9 ? '3.5' : '2.6'}
                    textAnchor="middle"
                    fontSize={n <= 9 ? '9' : '7'}
                    fontWeight="600"
                    fill="var(--text-primary)"
                    className="font-mono-num pointer-events-none"
                  >
                    {q.name.slice(0, 4)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="flex items-center justify-between text-[11px] text-[var(--text-secondary)]">
        <span>
          {isRydberg
            ? `拖动原子调整位置 · 阻塞半径 ${rb.toFixed(1)} μm`
            : hardware.topology_mode === 'custom'
            ? '依次点击两个比特以连接或断开耦合边'
            : `当前耦合边数: ${totalEdgeCount}`}
        </span>
        <span className="font-mono-num">{totalEdgeCount} 条耦合</span>
      </div>
    </div>
  );
};
