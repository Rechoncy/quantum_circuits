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

    if (validQ.length >= 2 && (uname === 'CZ' || uname === 'CNOT' || uname === 'CX' || uname === 'CCZ')) {
      const minQ = Math.min(...validQ);
      const maxQ = Math.max(...validQ);
      const yTop = topPad + minQ * rowH + rowH / 2;
      const yBot = topPad + maxQ * rowH + rowH / 2;
      parts.push(`<line x1="${cx}" y1="${yTop}" x2="${cx}" y2="${yBot}" stroke="#1E3A8A" stroke-width="2"/>`);
      const controls = uname === 'CZ' || uname === 'CCZ' ? validQ : validQ.slice(0, -1);
      for (const cq of controls) {
        const cy = topPad + cq * rowH + rowH / 2;
        parts.push(`<circle cx="${cx}" cy="${cy}" r="5.5" fill="#1E3A8A"/>`);
      }
      if (uname === 'CNOT' || uname === 'CX') {
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

export function downloadTextFile(filename: string, content: string, mime = 'image/svg+xml;charset=utf-8'): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
