# Quantum Studio v2 — 多后端量子线路演化求解与硬件感知最优综合工作台

<p align="center">
  <img src="assets/quantum_studio.png" width="96" alt="Quantum Studio Icon" />
</p>

一套集成了 **QuTiP（含时脉冲哈密顿量与主方程物理演化）**、**Kraus（通用门级密度矩阵演化）**、**Stim（稳定子追踪与百万级 Pauli 噪声采样）**、**硬件感知量子线路综合优化器（Euler / KAK Cartan 分解 / SABRE 拓扑路由）** 以及 **多轮对话式 AI 自然语言量子线路生成器** 的桌面与 Web 一体化工作台。

---

## 核心特性

1. **多尺度量子演化与噪声求解器 (`QuantumSolver`)**：
   - **`backend="qutip"`（实验物理模型级）**：基于 `qutip.mesolve` 求解含时脉冲哈密顿量 $H(t) = H_{\text{drift}} + \sum_j c_j(t) H_j$（支持余弦 Hann、高斯 Gaussian 与矩形包络及同层并行脉冲演化）与林德布拉德主方程（$T_1$ 弛豫、$T_2^*$ 纯退相干、静态 $ZZ$ 串扰与门内连续 Bloch 轨迹）。
   - **`backend="stim"`（高速稳定子与 Pauli 噪声采样级）**：基于 `stim.TableauSimulator` 输出每步稳定子生成元与 Bloch 矢量，并通过 **Pauli Twirling** 将 $T_1/T_2$ 与退极化噪声映射为 `PAULI_CHANNEL_1/2`。
   - **`backend="stim_qutip"`（QuTiP-Stim 物理桥接模式）**：先用 QuTiP 对每个物理量子门的含噪动力学做 Pauli Transfer Matrix (PTM) 标定提取等效 Pauli 错误率，再注入 Stim 高速采样。
   - **`backend="kraus"`（通用非 Clifford 密度矩阵级）**：支持任意参数旋转门、自定义矩阵/哈密顿量门等非 Clifford 门的精确 Kraus 算符演化。

2. **硬件感知量子线路设计与优化器 (`QuantumCircuitOptimizer`)**：
   - 支持预设及自定义实验平台原生门集（`NativeGateSet`）：
     - 中性原子里德堡阵列：`{RXY(θ, φ), Virtual-RZ (0 ns), Rydberg CZ}`（含二维光镊坐标与里德堡阻塞半径 $R_b$ 拓扑判定）
     - 超导 CZ 体系：`{Virtual-RZ (0 ns), SX, X, CZ}`
     - 超导 iSWAP 体系：`{Virtual-RZ (0 ns), SX, X, iSWAP}`
     - 超导 CNOT 体系：`{Virtual-RZ (0 ns), SX, X, CNOT}`
     - 离子阱 MS 体系：`{RXY(θ, φ), Virtual-RZ (0 ns), MS}`
   - **单比特最优分解 (`euler_1q.py`)** + **两比特 KAK / Cartan 最优综合 (`kak_2q.py`)** + **SABRE 启发式动态拓扑路由 (`passes.py`)** + **目标酉矩阵 / 泡利哈密顿量直出综合**。

3. **多轮对话式自然语言 → 量子线路助手 (`Ctrl+K`)**：
   - 支持中英文算法模板（Bell、GHZ、W 态、QFT、Grover、量子隐形传态、VQE、QAOA MaxCut、3Q 纠错码、Ramsey/Rabi）与自由分步门指令；
   - 支持多轮上下文对话精调（修改噪声率/参数、全局或按位置替换/删除量子门、增减比特并自动补齐光镊坐标、切换硬件平台）与历史版本一键回退；
   - 内置免密钥语义编译引擎，同时支持接入外部 OpenAI 兼容 LLM API（OpenAI / DeepSeek / Ollama）。

4. **60fps 渐进式脉冲演化动画与 3D 交互式 Bloch 球**：
   - 2× Retina Canvas 3D 玻璃质感 Bloch 球（支持拖拽旋转惯性、`3D / XY / XZ / YZ` 视角平滑旋转、高清 PNG 导出）；
   - 60fps 连续浮点时间轴（支持 `0.5× / 1× / 2×` 倍速、步进缓动、曲线与 3D 轨迹随时间游标从左向右实时生长绘制、当前执行门脉冲呼吸高亮）。

---

## 安装与快速启动

### 方式一：使用免配置 `.exe` 安装包（推荐）
直接从 [GitHub Releases](../../releases) 下载 **`QuantumStudio_v2.0.0_Setup.exe`**（已完整内嵌 Python 3.13、QuTiP、SciPy、Stim 及前端构建产物，无需本机安装 Python 或 Node.js）：
- 双击 `QuantumStudio_v2.0.0_Setup.exe`，选择安装目录即可自动释放并创建带专属图标的桌面与开始菜单快捷方式。

### 方式二：项目目录直接运行（已内嵌 `python_env/`）
```powershell
# 1. 直接双击原生桌面启动器（无命令行黑框，带专属图标）
.\QuantumStudio.exe

# 2. 或使用批处理 / 内嵌 Python 启动桌面窗口
.\start_studio.bat
.\python_env\python.exe run_studio.py

# 3. 以本地 Web 服务模式启动（浏览器访问 http://127.0.0.1:8765）
.\python_env\python.exe run_studio.py --web --open-browser --port 8765

# 4. 运行全套单元测试
.\python_env\python.exe -m pytest tests/test_solver_and_optimizer.py -v

# 5. 重新打包单文件 .exe 安装程序
.\python_env\python.exe scripts/build_installer.py
```
