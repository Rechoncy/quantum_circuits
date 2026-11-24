import type { CircuitOpModel, ProjectModel } from '../types';
import { BUILTIN_GATES } from '../constants';

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatParams(params: (number | string)[]): string {
  if (!params || params.length === 0) return '';
  return params
    .map((p) => {
      if (typeof p === 'string') return p;
      const r = p / Math.PI;
      if (Math.abs(r - 1) < 1e-3) return 'π';
      if (Math.abs(r + 1) < 1e-3) return '-π';
      if (Math.abs(r - 0.5) < 1e-3) return 'π/2';
      if (Math.abs(r + 0.5) < 1e-3) return '-π/2';
      if (Math.abs(r - 0.25) < 1e-3) return 'π/4';
      if (Math.abs(r + 0.25) < 1e-3) return '-π/4';
      return Number(p).toFixed(2);
    })
    .join(', ');
}

/**
 * Generate a crisp, publication-grade vector SVG representation of the quantum circuit.
 */
export function buildCircuitSvg(
  project: ProjectModel,
  operationsOverride?: CircuitOpModel[],
  title?: string
): string {
  const ops = operationsOverride ?? project.operations;
  const nQ = Math.max(1, project.qubits.length);
  const maxStep = ops.reduce((m, o) => Math.max(m, o.step), 0);
  const numCols = Math.max(3, maxStep + 1);

  const rowH = 54;
  const colW = 86;
  const leftPad = 104;
  const topPad = title ? 44 : 24;
  const rightPad = 36;
  const botPad = 24;

  const width = leftPad + numCols * colW + rightPad;
  const height = topPad + nQ * rowH + botPad;

  const builtinMap = new Map(BUILTIN_GATES.map((g) => [g.name.toUpperCase(), g]));
  const customMap = new Map(project.custom_gates.map((g) => [g.name.toUpperCase(), g]));

  const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<style>text { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "PingFang SC", sans-serif; } .mono { font-family: "SF Mono", "JetBrains Mono", Consolas, monospace; }</style>`,
    `<rect width="${width}" height="${height}" rx="12" fill="#FFFFFF" stroke="#E5E5EA" stroke-width="1"/>`,
  ];

  if (title) {
    parts.push(
      `<text x="20" y="26" font-size="13" font-weight="600" fill="#1D1D1F">${escapeXml(title)}</text>`
    );
  }

  // Qubit labels + horizontal wires
  for (let q = 0; q < nQ; q++) {
    const qSpec = project.qubits[q];
    const y = topPad + q * rowH + rowH / 2;
    const initStr = (qSpec?.init_state || '|0>').replace('>', '⟩');
    parts.push(
      `<circle cx="22" cy="${y}" r="3.5" fill="${escapeXml(qSpec?.color || '#0A84FF')}"/>`,
      `<text x="90" y="${y + 4}" text-anchor="end" font-size="12.5" font-weight="600" class="mono" fill="#1D1D1F">${escapeXml(qSpec?.name || `q${q}`)} <tspan fill="#6E6E73" font-weight="400">${escapeXml(initStr)}</tspan></text>`,
      `<line x1="${leftPad}" y1="${y}" x2="${width - rightPad}" y2="${y}" stroke="#9CA3AF" stroke-width="1.5"/>`
    );
  }

  // Operations
  for (const op of ops) {
    const validQ = op.qubits.filter((q) => q >= 0 && q < nQ);
    if (validQ.length === 0) continue;
    const cx = leftPad + op.step * colW + colW / 2;
    const uname = op.name.toUpperCase();
    const cg = customMap.get(uname);
    const bg = builtinMap.get(uname);
    const label = cg ? cg.display_label || cg.name : bg?.label || op.name;
    const pTxt = formatParams(op.params);

    if (op.kind === 'measure') {
      for (const qIdx of validQ) {
        const y = topPad + qIdx * rowH + rowH / 2;
        parts.push(
          `<rect x="${cx - 17}" y="${y - 15}" width="34" height="30" rx="7" fill="#F2F2F7" stroke="#C7C7CC" stroke-width="1.2"/>`,
          `<path d="M ${cx - 8} ${y + 5} A 8 8 0 0 1 ${cx + 8} ${y + 5}" fill="none" stroke="#3A3A3C" stroke-width="1.5"/>`,
          `<line x1="${cx}" y1="${y + 5}" x2="${cx + 5}" y2="${y - 5}" stroke="#3A3A3C" stroke-width="1.5" stroke-linecap="round"/>`
        );
      }
      continue;
    }

    if (
      validQ.length >= 2 &&
      (uname === 'CZ' ||
        uname === 'CNOT' ||
        uname === 'CX' ||
        uname === 'CCZ' ||
        uname === 'CCX' ||
        uname === 'TOFFOLI' ||
        uname === 'SWAP')
    ) {
      const minQ = Math.min(...validQ);
      const maxQ = Math.max(...validQ);
      const yTop = topPad + minQ * rowH + rowH / 2;
      const yBot = topPad + maxQ * rowH + rowH / 2;
      parts.push(`<line x1="${cx}" y1="${yTop}" x2="${cx}" y2="${yBot}" stroke="#1E3A8A" stroke-width="2"/>`);
      if (uname === 'SWAP') {
        for (const sq of validQ.slice(0, 2)) {
          const sy = topPad + sq * rowH + rowH / 2;
          parts.push(
            `<line x1="${cx - 6}" y1="${sy - 6}" x2="${cx + 6}" y2="${sy + 6}" stroke="#1E3A8A" stroke-width="2.2" stroke-linecap="round"/>`,
            `<line x1="${cx - 6}" y1="${sy + 6}" x2="${cx + 6}" y2="${sy - 6}" stroke="#1E3A8A" stroke-width="2.2" stroke-linecap="round"/>`
          );
        }
        continue;
      }
      const controls = uname === 'CZ' || uname === 'CCZ' ? validQ : validQ.slice(0, -1);
      for (const cq of controls) {
        const cy = topPad + cq * rowH + rowH / 2;
        parts.push(`<circle cx="${cx}" cy="${cy}" r="5.5" fill="#1E3A8A"/>`);
      }
      if (uname === 'CNOT' || uname === 'CX' || uname === 'CCX' || uname === 'TOFFOLI') {
        const ty = topPad + validQ[validQ.length - 1] * rowH + rowH / 2;
        parts.push(
          `<circle cx="${cx}" cy="${ty}" r="12" fill="#FFFFFF" stroke="#1E3A8A" stroke-width="2"/>`,
          `<line x1="${cx - 7}" y1="${ty}" x2="${cx + 7}" y2="${ty}" stroke="#1E3A8A" stroke-width="1.8"/>`,
          `<line x1="${cx}" y1="${ty - 7}" x2="${cx}" y2="${ty + 7}" stroke="#1E3A8A" stroke-width="1.8"/>`
        );
      } else {
        const midY = (yTop + yBot) / 2;
        parts.push(
          `<rect x="${cx + 8}" y="${midY - 8}" width="24" height="16" rx="4" fill="#EEF2FF" stroke="#C7D2FE"/>`,
          `<text x="${cx + 20}" y="${midY + 3.5}" text-anchor="middle" font-size="9.5" font-weight="700" class="mono" fill="#3730A3">${escapeXml(uname)}</text>`
        );
      }
      continue;
    }

    // Color scheme
    let fill = '#F0F9FF';
    let stroke = '#7DD3FC';
    let textCol = '#0C4A6E';
    let dash = '';
    if (cg) {
      fill = '#FAF5FF';
      stroke = '#D8B4FE';
      textCol = '#6B21A8';
    } else if (op.kind === 'noise') {
      fill = '#FFFBEB';
      stroke = '#FCD34D';
      textCol = '#92400E';
      dash = ' stroke-dasharray="4 3"';
    } else if (bg?.tint === 'teal') {
      fill = '#ECFDF5';
      stroke = '#6EE7B7';
      textCol = '#065F46';
      if (uname === 'RZ') dash = ' stroke-dasharray="4 3"';
    } else if (bg?.tint === 'indigo') {
      fill = '#EEF2FF';
      stroke = '#A5B4FC';
      textCol = '#3730A3';
    }

    if (validQ.length >= 2) {
      const minQ = Math.min(...validQ);
      const maxQ = Math.max(...validQ);
      const boxY = topPad + minQ * rowH + rowH / 2 - 19;
      const boxH = (maxQ - minQ) * rowH + 38;
      parts.push(
        `<rect x="${cx - 32}" y="${boxY}" width="64" height="${boxH}" rx="10" fill="${fill}" stroke="${stroke}" stroke-width="1.4"${dash}/>`,
        `<text x="${cx}" y="${boxY + boxH / 2 + (pTxt ? -2 : 4)}" text-anchor="middle" font-size="12" font-weight="700" class="mono" fill="${textCol}">${escapeXml(label)}</text>`
      );
      if (pTxt) {
        parts.push(
          `<text x="${cx}" y="${boxY + boxH / 2 + 12}" text-anchor="middle" font-size="9.5" class="mono" fill="${textCol}" opacity="0.85">${escapeXml(pTxt)}</text>`
        );
      }
    } else {
      const cy = topPad + validQ[0] * rowH + rowH / 2;
      const boxW = pTxt ? Math.min(76, Math.max(46, Math.round(pTxt.length * 6 + 14))) : 44;
      const boxH = 36;
      parts.push(
        `<rect x="${cx - boxW / 2}" y="${cy - boxH / 2}" width="${boxW}" height="${boxH}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1.3"${dash}/>`,
        `<text x="${cx}" y="${cy + (pTxt ? -2 : 4)}" text-anchor="middle" font-size="12" font-weight="700" class="mono" fill="${textCol}">${escapeXml(label)}</text>`
      );
      if (pTxt) {
        parts.push(
          `<text x="${cx}" y="${cy + 11}" text-anchor="middle" font-size="9.5" class="mono" fill="${textCol}" opacity="0.85">${escapeXml(pTxt)}</text>`
        );
      }
    }
  }

  parts.push('</svg>');
  return parts.join('\n');
}

function formatLatexParam(p: number | string): string {
  if (typeof p === 'string') {
    const greek: Record<string, string> = {
      theta: '\\theta',
      phi: '\\phi',
      gamma: '\\gamma',
      beta: '\\beta',
      alpha: '\\alpha',
      lambda: '\\lambda',
    };
    return greek[p.trim()] || p.replace(/_/g, '\\_');
  }
  const r = p / Math.PI;
  if (Math.abs(r - 1) < 1e-3) return '\\pi';
  if (Math.abs(r + 1) < 1e-3) return '-\\pi';
  if (Math.abs(r - 0.5) < 1e-3) return '\\pi/2';
  if (Math.abs(r + 0.5) < 1e-3) return '-\\pi/2';
  if (Math.abs(r - 0.25) < 1e-3) return '\\pi/4';
  if (Math.abs(r + 0.25) < 1e-3) return '-\\pi/4';
  return Number(p).toFixed(2);
}

/**
 * Generate publication-ready LaTeX `quantikz` source code for Overleaf / PRL / PRX Quantum papers.
 */
export function buildQuantikzLatex(project: ProjectModel): string {
  const nQ = Math.max(1, project.qubits.length);
  if (nQ > 40) {
    return `% 当前寄存器共 ${nQ} 个量子比特，超出单页论文排版上限 (≤40Q)，建议导出 OpenQASM 3.0 或使用子线路截图。\n`;
  }
  const ordered = [...project.operations].sort((a, b) => a.step - b.step);
  const maxStep = ordered.reduce((m, o) => Math.max(m, o.step), -1);
  const numCols = Math.max(1, maxStep + 1);

  const grid: string[][] = Array.from({ length: nQ }, () =>
    Array.from({ length: numCols }, () => '\\qw')
  );

  const texGateMap: Record<string, string> = {
    H: 'H',
    X: 'X',
    Y: 'Y',
    Z: 'Z',
    S: 'S',
    SDG: 'S^\\dagger',
    T: 'T',
    TDG: 'T^\\dagger',
    SX: '\\sqrt{X}',
    SXDG: '\\sqrt{X}^\\dagger',
    RX: 'R_x',
    RY: 'R_y',
    RZ: 'R_z',
    RXY: 'R_{xy}',
    RXX: 'R_{xx}',
    RYY: 'R_{yy}',
    RZZ: 'R_{zz}',
    MS: '\\text{MS}',
    ISWAP: '\\text{iSWAP}',
    SQISWAP: '\\sqrt{\\text{iSWAP}}',
    U3: 'U_3',
  };

  for (const op of ordered) {
    const validQ = op.qubits.filter((q) => q >= 0 && q < nQ);
    if (validQ.length === 0) continue;
    const s = Math.max(0, Math.min(numCols - 1, op.step));
    const u = op.name.toUpperCase();

    if (op.kind === 'measure') {
      for (const q of validQ) grid[q][s] = '\\meter{}';
      continue;
    }

    if ((u === 'CNOT' || u === 'CX') && validQ.length === 2) {
      const [ctrl, tgt] = validQ;
      grid[ctrl][s] = `\\ctrl{${tgt - ctrl}}`;
      grid[tgt][s] = '\\targ{}';
      continue;
    }
    if (u === 'CZ' && validQ.length === 2) {
      const [c, t] = validQ;
      grid[c][s] = `\\ctrl{${t - c}}`;
      grid[t][s] = '\\control{}';
      continue;
    }
    if (u === 'SWAP' && validQ.length === 2) {
      const [a, b] = validQ;
      grid[a][s] = `\\swap{${b - a}}`;
      grid[b][s] = '\\targX{}';
      continue;
    }
    if ((u === 'CCX' || u === 'TOFFOLI') && validQ.length === 3) {
      const [c0, c1, tgt] = validQ;
      grid[c0][s] = `\\ctrl{${tgt - c0}}`;
      grid[c1][s] = `\\ctrl{${tgt - c1}}`;
      grid[tgt][s] = '\\targ{}';
      continue;
    }
    if (u === 'CCZ' && validQ.length === 3) {
      const sorted = [...validQ].sort((a, b) => a - b);
      grid[sorted[0]][s] = `\\ctrl{${sorted[1] - sorted[0]}}`;
      grid[sorted[1]][s] = `\\ctrl{${sorted[2] - sorted[1]}}`;
      grid[sorted[2]][s] = '\\control{}';
      continue;
    }

    const baseTex = texGateMap[u] || `\\text{${op.name.replace(/_/g, '\\_')}}`;
    const pStr =
      op.params && op.params.length > 0
        ? `(${op.params.map(formatLatexParam).join(', ')})`
        : '';
    const fullLbl = `${baseTex}${pStr}`;

    if (validQ.length === 1) {
      grid[validQ[0]][s] = `\\gate{${fullLbl}}`;
    } else {
      const minQ = Math.min(...validQ);
      const maxQ = Math.max(...validQ);
      const span = maxQ - minQ + 1;
      grid[minQ][s] = `\\gate[wires=${span}]{${fullLbl}}`;
      for (let r = minQ + 1; r <= maxQ; r++) {
        grid[r][s] = '';
      }
    }
  }

  const lines: string[] = [
    '% LaTeX preamble: \\usepackage{tikz} \\usetikzlibrary{quantikz2}',
    '\\begin{quantikz}[row sep=0.35cm, column sep=0.45cm]',
  ];
  for (let q = 0; q < nQ; q++) {
    const qSpec = project.qubits[q];
    const ketRaw = (qSpec?.init_state || '|0>').replace(/^\|/, '').replace(/>$/, '');
    const qLabel = (qSpec?.name || `q_${q}`).replace(/_/g, '\\_');
    const rowStr = [`  \\lstick{$\\text{${qLabel}}\\ket{${ketRaw}}$}`, ...grid[q], '\\qw'].join(' & ');
    lines.push(q < nQ - 1 ? `${rowStr} \\\\` : rowStr);
  }
  lines.push('\\end{quantikz}');
  return lines.join('\n');
}

export function downloadTextFile(filename: string, content: string, mime = 'image/svg+xml;charset=utf-8'): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
