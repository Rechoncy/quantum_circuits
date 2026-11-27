# Legacy v1 PyQt6 GUI Archive (`v1.x`)

本目录归档了项目早期 `v1.x` 阶段的 **PyQt6 + Matplotlib 本地桌面界面与旧版启动器**，已由根目录新一代 **Quantum Studio EDA v2 (`QuantumStudio.exe` + `run_studio.py` + `web_studio/` + `quantum_solver/studio/`)** 完整取代。

## 归档文件清单
- `run_gui.py`：旧版 `v1.x` PyQt6 界面命令行入口脚本。
- `start_gui.bat`：旧版 `v1.x` Windows 批处理启动器。
- `app.py`：旧版 `QuantumWorkbenchWindow` 四分区 PyQt6 桌面窗口实现。
- `circuit_drawer.py`：旧版基于 Matplotlib 的静态量子线路绘图辅助模块。

> 注：纯 Python 量子线路 DSL 解析器（`circuit_parser.py`）与多维参数扫描引擎（`sweep_engine.py`）仍作为核心共享模块保留于 `quantum_solver/gui/` 中。
