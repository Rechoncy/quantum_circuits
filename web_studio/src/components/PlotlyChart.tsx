import React, { useEffect, useRef, useCallback, useState } from 'react';
import Plotly from 'plotly.js-dist-min';

export interface CursorTipPoint {
  y: number;
  color: string;
}

interface PlotlyChartProps {
  data: any[];
  layout: Record<string, any>;
  config?: Record<string, any>;
  /**
   * Optional 60fps x-axis oscilloscope probe playhead coordinate.
   * Rendered as a GPU-accelerated overlay synchronized with Plotly's axis geometry.
   */
  cursorX?: number | null;
  /**
   * Optional live readout badge displayed on the oscilloscope probe head (e.g. "S2 · 480ns").
   */
  cursorLabel?: string | null;
  /**
   * Optional interactive callback enabling direct click-and-drag scrubbing across the plot!
   */
  onScrubX?: (xVal: number) => void;
  /**
   * When true, progressively reveals the plot traces from x=0 up to `cursorX`
   * using Plotly's native SVG plot clip-rect.
   */
  progressiveReveal?: boolean;
  /**
   * Optional leading tip dots rendered right on the playhead at each curve's current y value.
   */
  cursorTips?: CursorTipPoint[];
  className?: string;
}

export const PlotlyChart: React.FC<PlotlyChartProps> = ({
  data,
  layout,
  config,
  cursorX = null,
  cursorLabel = null,
  onScrubX,
  progressiveReveal = false,
  cursorTips = [],
  className = 'w-full h-full',
}) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const cursorLineRef = useRef<HTMLDivElement>(null);
  const badgeRef = useRef<HTMLDivElement>(null);
  const tipDotsContainerRef = useRef<HTMLDivElement>(null);
  const [isScrubbing, setIsScrubbing] = useState(false);

  const cursorXRef = useRef<number | null>(cursorX);
  cursorXRef.current = cursorX;
  const progressiveRef = useRef<boolean>(progressiveReveal);
  progressiveRef.current = progressiveReveal;
  const cursorTipsRef = useRef<CursorTipPoint[]>(cursorTips);
  cursorTipsRef.current = cursorTips;
  const onScrubXRef = useRef(onScrubX);
  onScrubXRef.current = onScrubX;

  const clientXToDataX = useCallback((clientX: number): number | null => {
    const el = containerRef.current as any;
    if (!el) return null;
    const fullLayout = el._fullLayout;
    const xa = fullLayout?.xaxis;
    const sz = fullLayout?._size;
    if (!xa || !sz || typeof xa.p2c !== 'function') return null;
    const rect = el.getBoundingClientRect();
    const relPx = Math.max(0, Math.min(sz.w, clientX - rect.left - sz.l));
    const dataVal = Number(xa.p2c(relPx));
    return Number.isFinite(dataVal) ? dataVal : null;
  }, []);

  const updateCursorOverlay = useCallback(() => {
    const el = containerRef.current as any;
    const lineEl = cursorLineRef.current;
    const tipsEl = tipDotsContainerRef.current;
    const badgeEl = badgeRef.current;
    if (!el || !lineEl) return;

    const val = cursorXRef.current;
    const fullLayout = el._fullLayout;
    const xa = fullLayout?.xaxis;
    const ya = fullLayout?.yaxis;
    const sz = fullLayout?._size;

    const clipRects = el.querySelectorAll?.(
      'clipPath.plotclip > rect, clipPath[id$="xyplot"] > rect'
    ) as NodeListOf<SVGRectElement> | undefined;

    if (val === null || val === undefined || !xa || !sz || typeof xa.c2p !== 'function') {
      lineEl.style.opacity = '0';
      if (tipsEl) tipsEl.style.opacity = '0';
      if (clipRects) {
        clipRects.forEach((r) => {
          if (sz?.w) r.setAttribute('width', String(sz.w));
        });
      }
      return;
    }

    const relX = xa.c2p(val);
    if (!Number.isFinite(relX)) {
      lineEl.style.opacity = '0';
      if (tipsEl) tipsEl.style.opacity = '0';
      return;
    }

    const clampedRelX = Math.max(0, Math.min(sz.w, relX));
    const px = sz.l + clampedRelX;

    // Progressively clip the SVG plot traces so curves grow from left to right with the playhead
    if (clipRects && clipRects.length > 0) {
      const targetW = progressiveRef.current ? Math.max(0, clampedRelX) : sz.w;
      clipRects.forEach((r) => {
        r.setAttribute('width', String(targetW));
      });
    }

    lineEl.style.opacity = '1';
    lineEl.style.top = `${Math.max(2, sz.t - 6)}px`;
    lineEl.style.height = `${sz.h + 6}px`;
    lineEl.style.transform = `translate3d(${px}px, 0, 0)`;

    // Keep readout flag inside right boundary of plot
    if (badgeEl) {
      const nearRight = clampedRelX > sz.w - 86;
      badgeEl.style.transform = nearRight ? 'translateX(-100%) translateX(-4px)' : 'translateX(5px)';
    }

    // Position glowing leading tip dots on each curve at the playhead
    if (tipsEl && ya && typeof ya.c2p === 'function' && cursorTipsRef.current.length > 0) {
      tipsEl.style.opacity = '1';
      tipsEl.style.top = `${sz.t}px`;
      tipsEl.style.height = `${sz.h}px`;
      tipsEl.style.transform = `translate3d(${px}px, 0, 0)`;
      const children = tipsEl.children;
      const tips = cursorTipsRef.current;
      for (let i = 0; i < children.length; i++) {
        const dot = children[i] as HTMLElement;
        const tip = tips[i];
        if (!tip) {
          dot.style.opacity = '0';
          continue;
        }
        const relY = ya.c2p(tip.y);
        if (Number.isFinite(relY) && relY >= -6 && relY <= sz.h + 6) {
          dot.style.opacity = '1';
          dot.style.backgroundColor = tip.color;
          dot.style.boxShadow = `0 0 8px ${tip.color}`;
          dot.style.transform = `translate3d(-4.5px, ${relY - 4.5}px, 0)`;
        } else {
          dot.style.opacity = '0';
        }
      }
    } else if (tipsEl) {
      tipsEl.style.opacity = '0';
    }
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    Promise.resolve(
      Plotly.react(
        el,
        data,
        {
          autosize: true,
          dragmode: onScrubX ? false : undefined,
          font: {
            family:
              '"Segoe UI", "Inter", "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif',
            size: 11.5,
          },
          hoverlabel: {
            bgcolor: 'rgba(15, 20, 27, 0.94)',
            bordercolor: 'rgba(56, 70, 90, 0.9)',
            font: { color: '#E6EDF3', size: 11 },
          },
          ...layout,
        },
        {
          responsive: true,
          displaylogo: false,
          displayModeBar: true,
          modeBarButtonsToRemove: [
            'lasso2d',
            'select2d',
            'autoScale2d',
            'hoverClosestCartesian',
            'hoverCompareCartesian',
            'toggleSpikelines',
          ],
          ...config,
        }
      )
    ).then(() => {
      updateCursorOverlay();
    });
  }, [data, layout, config, Boolean(onScrubX), updateCursorOverlay]);

  useEffect(() => {
    updateCursorOverlay();
  }, [cursorX, cursorLabel, progressiveReveal, cursorTips, updateCursorOverlay]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      try {
        Plotly.Plots.resize(el);
        updateCursorOverlay();
      } catch {
        // ignore
      }
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      try {
        Plotly.purge(el);
      } catch {
        // ignore
      }
    };
  }, [updateCursorOverlay]);

  // Interactive click-and-drag waveform scrubbing when `onScrubX` is enabled
  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!onScrubXRef.current || e.button !== 0) return;
    // Ignore clicks on the top-right Plotly modebar buttons
    if ((e.target as HTMLElement)?.closest?.('.modebar')) return;
    const el = containerRef.current as any;
    const sz = el?._fullLayout?._size;
    if (sz) {
      const rect = el.getBoundingClientRect();
      const rx = e.clientX - rect.left;
      const ry = e.clientY - rect.top;
      if (rx < sz.l - 10 || rx > sz.l + sz.w + 10 || ry < sz.t - 14 || ry > sz.t + sz.h + 14) {
        return;
      }
    }
    const xVal = clientXToDataX(e.clientX);
    if (xVal === null) return;
    e.preventDefault();
    setIsScrubbing(true);
    onScrubXRef.current(xVal);

    const onMove = (ev: PointerEvent) => {
      const nextX = clientXToDataX(ev.clientX);
      if (nextX !== null && onScrubXRef.current) {
        onScrubXRef.current(nextX);
      }
    };
    const onUp = () => {
      setIsScrubbing(false);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  return (
    <div
      ref={wrapRef}
      onPointerDown={onScrubX ? handlePointerDown : undefined}
      className={`relative ${className} ${
        onScrubX ? (isScrubbing ? 'cursor-ew-resize select-none' : 'cursor-crosshair') : ''
      }`}
    >
      <div ref={containerRef} className="w-full h-full" />

      {/* Oscilloscope Probe Cursor Line + Triangle Probe Head + Optional Live Readout Tag */}
      <div
        ref={cursorLineRef}
        style={{ opacity: 0, left: 0 }}
        className="pointer-events-none absolute w-[1.5px] -ml-[0.75px] bg-[#2F81F7] shadow-[0_0_8px_rgba(47,129,247,0.65)] z-10 will-change-transform"
      >
        {/* Crisp Inverted-Triangle Oscilloscope Probe Marker ▼ */}
        <svg
          width="11"
          height="8"
          viewBox="0 0 11 8"
          className="absolute -left-[4.75px] -top-[2px] text-[#2F81F7] drop-shadow-xs"
        >
          <polygon points="0,0 11,0 5.5,8" fill="currentColor" />
        </svg>

        {cursorLabel && (
          <div
            ref={badgeRef}
            className="absolute top-0 left-0 px-1.5 py-0.25 rounded-[2px] bg-[#0f141b]/92 border border-[#2F81F7]/70 text-[#60a5fa] text-[9.5px] font-mono-num font-semibold whitespace-nowrap leading-tight shadow-xs"
          >
            {cursorLabel}
          </div>
        )}
      </div>

      {/* Leading Curve Intersection Dots */}
      <div
        ref={tipDotsContainerRef}
        style={{ opacity: 0, left: 0 }}
        className="pointer-events-none absolute top-0 left-0 w-0 z-20 will-change-transform"
      >
        {[0, 1, 2, 3, 4, 5, 6, 7].map((idx) => (
          <div
            key={idx}
            style={{ opacity: 0 }}
            className="absolute top-0 left-0 w-[9px] h-[9px] rounded-full ring-1.5 ring-white dark:ring-[#0f141b] will-change-transform"
          />
        ))}
      </div>
    </div>
  );
};
