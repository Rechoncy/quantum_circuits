import React from 'react';
import {
  Bot,
  CircuitBoard,
  Cpu,
  Gauge,
  Moon,
  PanelLeft,
  PanelRight,
  Sun,
} from 'lucide-react';
import type { PrimaryDomain } from '../types';

interface ActivityBarProps {
  domain: PrimaryDomain;
  onDomainChange: (d: PrimaryDomain) => void;
  showLeft: boolean;
  showRight: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  darkMode: boolean;
  onToggleDark: () => void;
  onOpenNlModal: () => void;
}

const DOMAIN_ITEMS: {
  id: PrimaryDomain;
  code: string;
  zh: string;
  title: string;
  icon: React.FC<{ className?: string }>;
}[] = [
  {
    id: 'circuits',
    code: 'CIR',
    zh: '线路',
    title: '量子线路工作区 (设计 / 仿真 / 综合 / 扫描)',
    icon: CircuitBoard,
  },
  {
    id: 'characterization',
    code: 'CAL',
    zh: '表征',
    title: '比特与门表征工作区 (Rabi / Ramsey / T1 / RB / AllXY / GST / XEB)',
    icon: Gauge,
  },
  {
    id: 'device',
    code: 'DEV',
    zh: '器件',
    title: '器件与硬件参数工作区 (拉比频率 / 上升下降沿 / 寄存器 / 拓扑)',
    icon: Cpu,
  },
];

export const ActivityBar: React.FC<ActivityBarProps> = ({
  domain,
  onDomainChange,
  showLeft,
  showRight,
  onToggleLeft,
  onToggleRight,
  darkMode,
  onToggleDark,
  onOpenNlModal,
}) => {
  return (
    <nav
      aria-label="IDE 三大平行主工作区导航栏"
      className="w-[52px] shrink-0 bg-[var(--bg-activity)] border-r border-[var(--border-hairline)] flex flex-col items-center justify-between py-1.5 select-none z-20"
    >
      {/* Top 3 Parallel First-Class Domains */}
      <div className="w-full flex flex-col items-center gap-0.5">
        {DOMAIN_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = domain === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onDomainChange(item.id)}
              title={item.title}
              className={`relative w-full h-[52px] flex flex-col items-center justify-center gap-1 transition-colors cursor-pointer ${
                active
                  ? 'text-[var(--accent)] bg-[var(--bg-sidebar)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.04]'
              }`}
            >
              {active && (
                <span className="absolute left-0 top-1 bottom-1 w-[3px] rounded-r-[1px] bg-[var(--accent)]" />
              )}
              <Icon className="w-[18px] h-[18px] stroke-[1.85]" />
              <span className="text-[10px] font-semibold leading-none tracking-tight">
                {item.zh}
              </span>
            </button>
          );
        })}

        <div className="w-7 my-1.5 border-t border-[var(--border-hairline)]" />

        {/* AI Circuit Synthesis Assistant */}
        <button
          onClick={onOpenNlModal}
          title="AI 量子线路综合助手 (Ctrl+K)"
          className="w-[38px] h-[38px] rounded-[3px] text-[var(--text-secondary)] hover:text-[var(--accent)] hover:bg-[var(--accent-soft)] flex flex-col items-center justify-center gap-0.5 transition-colors cursor-pointer"
        >
          <Bot className="w-[16px] h-[16px] stroke-[1.8]" />
          <span className="text-[8.5px] font-mono-num font-bold leading-none">AI</span>
        </button>
      </div>

      {/* Bottom Panel & Theme Controls */}
      <div className="w-full flex flex-col items-center gap-1 pb-1">
        {domain === 'circuits' && (
          <button
            onClick={onToggleLeft}
            title={showLeft ? '折叠左侧工程资源栏' : '展开左侧工程资源栏'}
            className={`w-[34px] h-[30px] rounded-[3px] flex items-center justify-center transition-colors cursor-pointer ${
              showLeft
                ? 'text-[var(--text-primary)] bg-black/[0.06] dark:bg-white/[0.08]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            <PanelLeft className="w-4 h-4" />
          </button>
        )}

        {domain !== 'device' && (
          <button
            onClick={onToggleRight}
            title={showRight ? '折叠右侧属性检查器' : '展开右侧属性检查器'}
            className={`w-[34px] h-[30px] rounded-[3px] flex items-center justify-center transition-colors cursor-pointer ${
              showRight
                ? 'text-[var(--text-primary)] bg-black/[0.06] dark:bg-white/[0.08]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            <PanelRight className="w-4 h-4" />
          </button>
        )}

        <button
          onClick={onToggleDark}
          title={darkMode ? '切换至浅色原理图主题' : '切换至深色工业工程主题'}
          className="w-[34px] h-[30px] rounded-[3px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.06] dark:hover:bg-white/[0.08] flex items-center justify-center transition-colors cursor-pointer"
        >
          {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
    </nav>
  );
};
