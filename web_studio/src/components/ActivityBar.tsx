import React from 'react';
import {
  Activity,
  Bot,
  CircuitBoard,
  Cpu,
  Gauge,
  LineChart,
  Moon,
  PanelLeft,
  PanelRight,
  Sun,
} from 'lucide-react';
import type { WorkMode } from '../types';

interface ActivityBarProps {
  mode: WorkMode;
  onModeChange: (m: WorkMode) => void;
  showLeft: boolean;
  showRight: boolean;
  onToggleLeft: () => void;
  onToggleRight: () => void;
  darkMode: boolean;
  onToggleDark: () => void;
  onOpenNlModal: () => void;
}

const NAV_ITEMS: {
  id: WorkMode;
  code: string;
  title: string;
  icon: React.FC<{ className?: string }>;
}[] = [
  { id: 'design', code: 'SCH', title: '原理图设计 (Design)', icon: CircuitBoard },
  { id: 'simulate', code: 'SIM', title: '量子演化仿真 (Simulate)', icon: Activity },
  { id: 'optimize', code: 'SYN', title: '硬件编译综合 (Optimize)', icon: Cpu },
  { id: 'sweep', code: 'SWP', title: '参数扫描 & VQE (Sweep)', icon: LineChart },
  { id: 'benchmark', code: 'CAL', title: '门标定基准 RB / AllXY / GST (Benchmark)', icon: Gauge },
];

export const ActivityBar: React.FC<ActivityBarProps> = ({
  mode,
  onModeChange,
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
      aria-label="IDE 主工作模式活动栏"
      className="w-[48px] shrink-0 bg-[var(--bg-activity)] border-r border-[var(--border-hairline)] flex flex-col items-center justify-between py-1 select-none z-20"
    >
      {/* Top Primary Workbench Modes */}
      <div className="w-full flex flex-col items-center">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = mode === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onModeChange(item.id)}
              title={item.title}
              className={`relative w-full h-[46px] flex flex-col items-center justify-center gap-0.5 transition-colors cursor-pointer ${
                active
                  ? 'text-[var(--accent)] bg-[var(--bg-sidebar)]'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-black/[0.04] dark:hover:bg-white/[0.04]'
              }`}
            >
              {active && (
                <span className="absolute left-0 top-0 bottom-0 w-[2.5px] bg-[var(--accent)]" />
              )}
              <Icon className="w-[17px] h-[17px] stroke-[1.8]" />
              <span className="text-[9px] font-mono-num font-bold tracking-wider leading-none">
                {item.code}
              </span>
            </button>
          );
        })}

        <div className="w-6 my-1.5 border-t border-[var(--border-hairline)]" />

        {/* AI Circuit Synthesis Assistant */}
        <button
          onClick={onOpenNlModal}
          title="AI 量子线路综合助手 (Ctrl+K)"
          className="w-[36px] h-[36px] rounded-[3px] text-[var(--text-secondary)] hover:text-[var(--accent)] hover:bg-[var(--accent-soft)] flex flex-col items-center justify-center gap-0.5 transition-colors cursor-pointer"
        >
          <Bot className="w-[16px] h-[16px] stroke-[1.8]" />
          <span className="text-[8.5px] font-mono-num font-bold leading-none">AI</span>
        </button>
      </div>

      {/* Bottom Panel & Theme Controls */}
      <div className="w-full flex flex-col items-center gap-1 pb-1">
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
