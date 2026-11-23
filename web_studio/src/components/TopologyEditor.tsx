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

  // Compute active edges based on topology_mode
  const edges: [number, number][] = [];
  for (let i = 0; i < qubits.length; i++) {
    for (let j = i + 1; j < qubits.length; j++) {
      if (hardware.topology_mode === 'all') {
        edges.push([i, j]);
      } else if (hardware.topology_mode === 'linear' && j === i + 1) {
        edges.push([i, j]);
      } else if (hardware.topology_mode === 'rydberg_radius') {
        const d = Math.hypot(qubits[i].pos_x - qubits[j].pos_x, qubits[i].pos_y - qubits[j].pos_y);
        if (d <= rb + 1e-6) edges.push([i, j]);
      } else if (hardware.topology_mode === 'custom') {
        if (hardware.edges.some(([u, v]) => (u === i && v === j) || (u === j && v === i))) {
          edges.push([i, j]);
        }
      }
    }
  }

  // Map μm coordinates (-4..20, -10..10) into SVG viewBox (0..260, 0..170)
  const scale = 11;
  const ox = 36;
  const oy = 85;
  const toSvg = (x: number, y: number) => ({ sx: ox + x * scale, sy: oy - y * scale });
  const fromSvg = (sx: number, sy: number) => ({
    x: Number(((sx - ox) / scale).toFixed(1)),
    y: Number(((oy - sy) / scale).toFixed(1)),
  });

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

          {/* Rydberg blockade radius rings */}
          {isRydberg &&
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
                strokeWidth="1.8"
                strokeOpacity="0.7"
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
                <circle
                  r={11}
                  fill="var(--bg-panel)"
                  stroke={isSel ? '#0A84FF' : q.color}
                  strokeWidth={isSel ? 2.5 : 2}
                />
                <text
                  y="3.5"
                  textAnchor="middle"
                  fontSize="9"
                  fontWeight="600"
                  fill="var(--text-primary)"
                  className="font-mono-num pointer-events-none"
                >
                  {q.name.slice(0, 4)}
                </text>
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
            : `当前耦合边数: ${edges.length}`}
        </span>
        <span className="font-mono-num">{edges.length} 条耦合</span>
      </div>
    </div>
  );
};
