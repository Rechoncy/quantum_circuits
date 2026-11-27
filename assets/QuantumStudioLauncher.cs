using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows.Forms;

[assembly: AssemblyTitle("Quantum Studio EDA")]
[assembly: AssemblyDescription("Quantum Studio EDA v2.8 — Quantum Circuit, Gate Characterization & Device Workbench")]
[assembly: AssemblyProduct("Quantum Studio EDA")]
[assembly: AssemblyVersion("2.8.0.0")]
[assembly: AssemblyFileVersion("2.8.0.0")]

namespace QuantumStudioLauncher
{
    internal static class Program
    {
        [DllImport("shell32.dll", SetLastError = true)]
        private static extern void SetCurrentProcessExplicitAppUserModelID([MarshalAs(UnmanagedType.LPWStr)] string AppID);

        [STAThread]
        private static int Main(string[] args)
        {
            try
            {
                SetCurrentProcessExplicitAppUserModelID("QuantumStudio.Desktop.v2");
            }
            catch
            {
                // Ignore on older Windows builds
            }

            string rootDir = AppDomain.CurrentDomain.BaseDirectory;
            string pyEnvDir = Path.Combine(rootDir, "python_env");
            string pyBinDir = Path.Combine(pyEnvDir, "Library", "bin");
            string runScript = Path.Combine(rootDir, "run_studio.py");

            bool consoleMode = false;
            foreach (string a in args)
            {
                if (a == "--web" || a == "--help" || a == "-h")
                {
                    consoleMode = true;
                    break;
                }
            }

            // Use pythonw.exe for GUI mode (already a Windows GUI subsystem executable, so no console window appears
            // and CreateNoWindow MUST remain false so Windows passes SW_SHOWNORMAL to the main GUI window).
            string pyExe = Path.Combine(pyEnvDir, consoleMode ? "python.exe" : "pythonw.exe");
            if (!File.Exists(pyExe))
            {
                pyExe = Path.Combine(pyEnvDir, "python.exe");
            }
            if (!File.Exists(pyExe))
            {
                MessageBox.Show(
                    "未找到项目内嵌 Python 运行时: " + pyExe,
                    "Quantum Studio 启动错误",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }

            string quotedScript = "\"" + runScript + "\"";
            string joinedArgs = args.Length > 0 ? " " + string.Join(" ", args) : "";

            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pyExe,
                Arguments = quotedScript + joinedArgs,
                WorkingDirectory = rootDir,
                UseShellExecute = false,
                CreateNoWindow = false,
                WindowStyle = ProcessWindowStyle.Normal
            };

            string oldPath = Environment.GetEnvironmentVariable("PATH") ?? "";
            psi.EnvironmentVariables["PATH"] = pyEnvDir + ";" + pyBinDir + ";" + oldPath;
            psi.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            psi.EnvironmentVariables["MPMATH_NOGMPY"] = "1";

            try
            {
                Process proc = Process.Start(psi);
                return proc != null ? 0 : 1;
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "启动 Quantum Studio 失败:\n" + ex.Message,
                    "Quantum Studio",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }
        }
    }
}
