import React, { useEffect, useRef, useCallback } from 'react';
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
   * Optional 60fps x-axis playhead coordinate.
   * Rendered as a GPU-accelerated overlay synchronized with Plotly's axis geometry
   * so animating at 60fps never triggers full SVG re-layout.
   */
  cursorX?: number | null;
  /**
   * When true, progressively reveals the plot traces from x=0 up to `cursorX`
   * using Plotly's native SVG plot clip-rect so future un-evolved curves are hidden
   * until the playhead sweeps across them.
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
  progressiveReveal = false,
  cursorTips = [],
  className = 'w-full h-full',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cursorLineRef = useRef<HTMLDivElement>(null);
  const tipDotsContainerRef = useRef<HTMLDivElement>(null);

  const cursorXRef = useRef<number | null>(cursorX);
  cursorXRef.current = cursorX;
  const progressiveRef = useRef<boolean>(progressiveReveal);
  progressiveRef.current = progressiveReveal;
  const cursorTipsRef = useRef<CursorTipPoint[]>(cursorTips);
  cursorTipsRef.current = cursorTips;

  const updateCursorOverlay = useCallback(() => {
    const el = containerRef.current as any;
    const lineEl = cursorLineRef.current;
    const tipsEl = tipDotsContainerRef.current;
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
    lineEl.style.top = `${sz.t}px`;
    lineEl.style.height = `${sz.h}px`;
    lineEl.style.transform = `translate3d(${px}px, 0, 0)`;

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
          dot.style.transform = `translate3d(-5px, ${relY - 5}px, 0)`;
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
          font: {
            family:
              '-apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", sans-serif',
            size: 11.5,
          },
          hoverlabel: {
            bgcolor: 'rgba(28, 28, 30, 0.92)',
            bordercolor: 'rgba(255,255,255,0.12)',
            font: { color: '#F5F5F7', size: 11.5 },
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
  }, [data, layout, config, updateCursorOverlay]);

  // Fast 60fps overlay & progressive clip-rect update whenever cursorX or tips change
  useEffect(() => {
    updateCursorOverlay();
  }, [cursorX, progressiveReveal, cursorTips, updateCursorOverlay]);

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

  return (
    <div className={`relative ${className}`}>
      <div ref={containerRef} className="w-full h-full" />
      <div
        ref={cursorLineRef}
        style={{ opacity: 0, left: 0 }}
        className="pointer-events-none absolute w-[2px] -ml-[1px] bg-[#0A84FF] shadow-[0_0_8px_rgba(10,132,255,0.55)] z-10 will-change-transform"
      >
        <div className="w-2.5 h-2.5 rounded-full bg-[#0A84FF] ring-2 ring-white dark:ring-[#222226] -ml-[4px] -mt-1 shadow-xs" />
      </div>
      <div
        ref={tipDotsContainerRef}
        style={{ opacity: 0, left: 0 }}
        className="pointer-events-none absolute top-0 left-0 w-0 z-20 will-change-transform"
      >
        {[0, 1, 2, 3, 4, 5, 6, 7].map((idx) => (
          <div
            key={idx}
            style={{ opacity: 0 }}
            className="absolute top-0 left-0 w-[10px] h-[10px] rounded-full ring-2 ring-white dark:ring-[#222226] will-change-transform"
          />
        ))}
      </div>
    </div>
  );
};
