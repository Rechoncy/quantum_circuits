using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyTitle("Quantum Studio v2.3.0 Setup")]
[assembly: AssemblyDescription("Quantum Studio v2.3.0 Self-Contained Desktop Installer & Uninstaller (Embedded Python 3.13 + QuTiP + Trajectory + Stim)")]
[assembly: AssemblyProduct("Quantum Studio v2")]
[assembly: AssemblyVersion("2.3.0.0")]
[assembly: AssemblyFileVersion("2.3.0.0")]

namespace QuantumStudioSetup
{
    /// <summary>
    /// Read-only stream slice over the appended ZIP payload inside the Setup .exe binary.
    /// </summary>
    internal sealed class SubStream : Stream
    {
        private readonly Stream _baseStream;
        private readonly long _offset;
        private readonly long _length;
        private long _pos;

        public SubStream(Stream baseStream, long offset, long length)
        {
            _baseStream = baseStream;
            _offset = offset;
            _length = length;
            _pos = 0;
            _baseStream.Seek(_offset, SeekOrigin.Begin);
        }

        public override bool CanRead { get { return true; } }
        public override bool CanSeek { get { return true; } }
        public override bool CanWrite { get { return false; } }
        public override long Length { get { return _length; } }

        public override long Position
        {
            get { return _pos; }
            set { Seek(value, SeekOrigin.Begin); }
        }

        public override void Flush() { }

        public override int Read(byte[] buffer, int offset, int count)
        {
            long remaining = _length - _pos;
            if (remaining <= 0) return 0;
            if (count > remaining) count = (int)remaining;
            _baseStream.Seek(_offset + _pos, SeekOrigin.Begin);
            int read = _baseStream.Read(buffer, offset, count);
            _pos += read;
            return read;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            long target = _pos;
            if (origin == SeekOrigin.Begin) target = offset;
            else if (origin == SeekOrigin.Current) target = _pos + offset;
            else if (origin == SeekOrigin.End) target = _length + offset;

            if (target < 0) target = 0;
            if (target > _length) target = _length;
            _pos = target;
            _baseStream.Seek(_offset + _pos, SeekOrigin.Begin);
            return _pos;
        }

        public override void SetLength(long value) { throw new NotSupportedException(); }
        public override void Write(byte[] buffer, int offset, int count) { throw new NotSupportedException(); }
    }

    internal sealed class SetupForm : Form
    {
        private readonly TextBox _dirBox;
        private readonly Button _browseBtn;
        private readonly CheckBox _chkDesktop;
        private readonly CheckBox _chkStartMenu;
        private readonly CheckBox _chkLaunch;
        private readonly ProgressBar _progressBar;
        private readonly Label _statusLabel;
        private readonly Button _installBtn;
        private readonly Button _cancelBtn;
        private bool _installing;

        public SetupForm(string defaultDir)
        {
            Text = "Quantum Studio v2.3.0 安装程序";
            Size = new Size(560, 395);
            MinimumSize = new Size(560, 395);
            MaximumSize = new Size(560, 395);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            BackColor = Color.FromArgb(245, 245, 247);
            Font = new Font("Microsoft YaHei UI", 9.5f, FontStyle.Regular);

            try
            {
                Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            }
            catch { }

            // Header Banner
            Panel header = new Panel
            {
                Dock = DockStyle.Top,
                Height = 76,
                BackColor = Color.FromArgb(15, 34, 68)
            };

            PictureBox logoBox = new PictureBox
            {
                Location = new Point(20, 14),
                Size = new Size(48, 48),
                SizeMode = PictureBoxSizeMode.StretchImage
            };
            try
            {
                if (Icon != null) logoBox.Image = Icon.ToBitmap();
            }
            catch { }
            header.Controls.Add(logoBox);

            Label titleLbl = new Label
            {
                Text = "Quantum Studio v2.3.0",
                ForeColor = Color.White,
                Font = new Font("Microsoft YaHei UI", 13f, FontStyle.Bold),
                Location = new Point(80, 14),
                AutoSize = true
            };
            header.Controls.Add(titleLbl);

            Label subLbl = new Label
            {
                Text = "500+Q 多后端量子演化求解、OpenQASM 3 双向编译、变分能量景观与 AI 线路工作台",
                ForeColor = Color.FromArgb(186, 230, 253),
                Font = new Font("Microsoft YaHei UI", 8.5f, FontStyle.Regular),
                Location = new Point(82, 43),
                AutoSize = true
            };
            header.Controls.Add(subLbl);
            Controls.Add(header);

            // Install path section
            Label dirTitle = new Label
            {
                Text = "目标安装目录：",
                Location = new Point(24, 94),
                AutoSize = true,
                Font = new Font("Microsoft YaHei UI", 9.5f, FontStyle.Bold)
            };
            Controls.Add(dirTitle);

            _dirBox = new TextBox
            {
                Text = defaultDir,
                Location = new Point(24, 118),
                Size = new Size(412, 26),
                Font = new Font("Consolas", 9.5f, FontStyle.Regular)
            };
            Controls.Add(_dirBox);

            _browseBtn = new Button
            {
                Text = "浏览...",
                Location = new Point(444, 116),
                Size = new Size(84, 28),
                FlatStyle = FlatStyle.System
            };
            _browseBtn.Click += OnBrowseClick;
            Controls.Add(_browseBtn);

            // Options
            _chkDesktop = new CheckBox
            {
                Text = "在桌面创建 Quantum Studio 快捷方式（含专属图标）",
                Checked = true,
                Location = new Point(26, 160),
                AutoSize = true
            };
            Controls.Add(_chkDesktop);

            _chkStartMenu = new CheckBox
            {
                Text = "在开始菜单程序列表创建快捷方式",
                Checked = true,
                Location = new Point(26, 188),
                AutoSize = true
            };
            Controls.Add(_chkStartMenu);

            _chkLaunch = new CheckBox
            {
                Text = "安装完成后立即启动 Quantum Studio",
                Checked = true,
                Location = new Point(26, 216),
                AutoSize = true
            };
            Controls.Add(_chkLaunch);

            // Progress area
            _statusLabel = new Label
            {
                Text = "准备就绪，点击「立即安装」开始释放文件。",
                Location = new Point(24, 254),
                Size = new Size(504, 20),
                ForeColor = Color.FromArgb(90, 90, 96)
            };
            Controls.Add(_statusLabel);

            _progressBar = new ProgressBar
            {
                Location = new Point(24, 278),
                Size = new Size(504, 20),
                Minimum = 0,
                Maximum = 1000,
                Value = 0
            };
            Controls.Add(_progressBar);

            // Bottom buttons
            _installBtn = new Button
            {
                Text = "立即安装",
                Location = new Point(332, 316),
                Size = new Size(100, 32),
                BackColor = Color.FromArgb(0, 113, 227),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat
            };
            _installBtn.FlatAppearance.BorderSize = 0;
            _installBtn.Click += OnInstallClick;
            Controls.Add(_installBtn);

            _cancelBtn = new Button
            {
                Text = "取消",
                Location = new Point(440, 316),
                Size = new Size(88, 32),
                FlatStyle = FlatStyle.System
            };
            _cancelBtn.Click += (s, e) => { if (!_installing) Close(); };
            Controls.Add(_cancelBtn);
        }

        private void OnBrowseClick(object sender, EventArgs e)
        {
            using (FolderBrowserDialog fbd = new FolderBrowserDialog())
            {
                fbd.Description = "选择 Quantum Studio v2 安装目录";
                fbd.SelectedPath = _dirBox.Text;
                if (fbd.ShowDialog(this) == DialogResult.OK)
                {
                    _dirBox.Text = fbd.SelectedPath;
                }
            }
        }

        private void OnInstallClick(object sender, EventArgs e)
        {
            if (_installing) return;
            string targetDir = _dirBox.Text.Trim();
            if (string.IsNullOrEmpty(targetDir))
            {
                MessageBox.Show(this, "请选择有效的安装目录。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            _installing = true;
            _dirBox.Enabled = false;
            _browseBtn.Enabled = false;
            _chkDesktop.Enabled = false;
            _chkStartMenu.Enabled = false;
            _installBtn.Enabled = false;
            _cancelBtn.Enabled = false;

            bool makeDesktop = _chkDesktop.Checked;
            bool makeStartMenu = _chkStartMenu.Checked;

            Thread worker = new Thread(() =>
            {
                try
                {
                    InstallerEngine.ExtractPayload(
                        targetDir,
                        makeDesktop,
                        makeStartMenu,
                        (pct, msg) =>
                        {
                            try
                            {
                                BeginInvoke((Action)(() =>
                                {
                                    _progressBar.Value = Math.Max(0, Math.Min(1000, pct));
                                    _statusLabel.Text = msg;
                                }));
                            }
                            catch { }
                        }
                    );

                    BeginInvoke((Action)(() =>
                    {
                        _progressBar.Value = 1000;
                        _statusLabel.Text = "安装完成！";
                        bool launchNow = _chkLaunch.Checked;
                        MessageBox.Show(
                            this,
                            "Quantum Studio v2.3.0 已成功安装至:\n" + targetDir + "\n\n（如需卸载，可在 Windows「设置 → 已安装的应用」或安装目录运行 Uninstall.exe）",
                            "安装成功",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Information
                        );
                        if (launchNow)
                        {
                            string exePath = Path.Combine(targetDir, "QuantumStudio.exe");
                            if (File.Exists(exePath))
                            {
                                Process.Start(new ProcessStartInfo
                                {
                                    FileName = exePath,
                                    WorkingDirectory = targetDir,
                                    UseShellExecute = true
                                });
                            }
                        }
                        _installing = false;
                        Close();
                    }));
                }
                catch (Exception ex)
                {
                    BeginInvoke((Action)(() =>
                    {
                        _installing = false;
                        _dirBox.Enabled = true;
                        _browseBtn.Enabled = true;
                        _chkDesktop.Enabled = true;
                        _chkStartMenu.Enabled = true;
                        _installBtn.Enabled = true;
                        _cancelBtn.Enabled = true;
                        _statusLabel.Text = "安装出错: " + ex.Message;
                        MessageBox.Show(
                            this,
                            "安装过程中出现错误:\n" + ex.Message,
                            "安装错误",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Error
                        );
                    }));
                }
            });
            worker.IsBackground = true;
            worker.Start();
        }
    }

    internal static class InstallerEngine
    {
        private const string MAGIC = "QSTUDIO2";
        internal const string UNINSTALL_REG_KEY = @"Software\Microsoft\Windows\CurrentVersion\Uninstall\QuantumStudio";

        public static bool FindAppendedPayload(FileStream fs, out long offset, out long length)
        {
            offset = 0;
            length = 0;
            if (fs.Length < 24) return false;
            fs.Seek(fs.Length - 24, SeekOrigin.Begin);
            BinaryReader br = new BinaryReader(fs);
            long off = br.ReadInt64();
            long len = br.ReadInt64();
            byte[] magicBytes = br.ReadBytes(8);
            string magic = Encoding.ASCII.GetString(magicBytes);
            if (magic != MAGIC) return false;
            if (off < 0 || len <= 0 || off + len + 24 > fs.Length) return false;
            offset = off;
            length = len;
            return true;
        }

        public static void ExtractPayload(
            string targetDir,
            bool createDesktopShortcut,
            bool createStartMenuShortcut,
            Action<int, string> onProgress)
        {
            string selfExe = Application.ExecutablePath;
            long totalExtractedBytes = 0;
            byte[] stubBytes = null;
            using (FileStream fs = new FileStream(selfExe, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                long offset, length;
                if (!FindAppendedPayload(fs, out offset, out length))
                {
                    throw new InvalidOperationException("安装包未包含内嵌数据负载 (QSTUDIO2 footer missing)。");
                }

                // Read pure WinForms stub [0..offset) to emit standalone Uninstall.exe (~28 KB)
                if (offset > 0 && offset < 2 * 1024 * 1024)
                {
                    fs.Seek(0, SeekOrigin.Begin);
                    stubBytes = new byte[(int)offset];
                    int got = 0;
                    while (got < stubBytes.Length)
                    {
                        int n = fs.Read(stubBytes, got, stubBytes.Length - got);
                        if (n <= 0) break;
                        got += n;
                    }
                }

                Directory.CreateDirectory(targetDir);
                using (SubStream sub = new SubStream(fs, offset, length))
                using (ZipArchive archive = new ZipArchive(sub, ZipArchiveMode.Read))
                {
                    int total = Math.Max(1, archive.Entries.Count);
                    int current = 0;
                    foreach (ZipArchiveEntry entry in archive.Entries)
                    {
                        current++;
                        string relPath = entry.FullName.Replace('/', Path.DirectorySeparatorChar);
                        string destPath = Path.Combine(targetDir, relPath);

                        if (string.IsNullOrEmpty(entry.Name))
                        {
                            Directory.CreateDirectory(destPath);
                            continue;
                        }

                        string parentDir = Path.GetDirectoryName(destPath);
                        if (!string.IsNullOrEmpty(parentDir))
                        {
                            Directory.CreateDirectory(parentDir);
                        }

                        entry.ExtractToFile(destPath, true);
                        totalExtractedBytes += entry.Length;

                        if (onProgress != null && (current % 25 == 0 || current == total))
                        {
                            int permille = (int)((long)current * 975 / total);
                            onProgress(permille, string.Format("正在安装 ({0}/{1}): {2}", current, total, entry.Name));
                        }
                    }
                }
            }

            string appExe = Path.Combine(targetDir, "QuantumStudio.exe");
            string uninstallExe = Path.Combine(targetDir, "Uninstall.exe");
            string iconPath = Path.Combine(targetDir, "assets", "quantum_studio.ico");

            if (stubBytes != null && stubBytes.Length > 0)
            {
                try
                {
                    File.WriteAllBytes(uninstallExe, stubBytes);
                }
                catch { }
            }

            if (createDesktopShortcut || createStartMenuShortcut)
            {
                RegisterUninstaller(targetDir, appExe, uninstallExe, iconPath, totalExtractedBytes);
            }

            if (createDesktopShortcut)
            {
                if (onProgress != null) onProgress(990, "正在创建桌面快捷方式...");
                string desktopDir = Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
                CreateShortcut(Path.Combine(desktopDir, "Quantum Studio.lnk"), appExe, targetDir, iconPath);
            }

            if (createStartMenuShortcut)
            {
                if (onProgress != null) onProgress(995, "正在创建开始菜单快捷方式...");
                string programsDir = Environment.GetFolderPath(Environment.SpecialFolder.Programs);
                CreateShortcut(Path.Combine(programsDir, "Quantum Studio.lnk"), appExe, targetDir, iconPath);
            }

            if (onProgress != null) onProgress(1000, "安装完成！");
        }

        private static void RegisterUninstaller(
            string targetDir,
            string appExe,
            string uninstallExe,
            string iconPath,
            long sizeBytes)
        {
            try
            {
                using (RegistryKey key = Registry.CurrentUser.CreateSubKey(UNINSTALL_REG_KEY))
                {
                    if (key == null) return;
                    key.SetValue("DisplayName", "Quantum Studio v2.3.0");
                    key.SetValue("DisplayVersion", "2.3.0");
                    key.SetValue("Publisher", "Quantum Studio Project");
                    key.SetValue("InstallLocation", targetDir);
                    key.SetValue("DisplayIcon", File.Exists(iconPath) ? iconPath : appExe);
                    key.SetValue("UninstallString", "\"" + uninstallExe + "\" --uninstall");
                    key.SetValue("QuietUninstallString", "\"" + uninstallExe + "\" --uninstall /S");
                    key.SetValue("NoModify", 1, RegistryValueKind.DWord);
                    key.SetValue("NoRepair", 1, RegistryValueKind.DWord);
                    int estKb = (int)Math.Max(1024L, sizeBytes / 1024L);
                    key.SetValue("EstimatedSize", estKb, RegistryValueKind.DWord);
                }
            }
            catch { }
        }

        public static int RunUninstall(string explicitDir, bool silent)
        {
            string selfExe = Application.ExecutablePath;
            string targetDir = explicitDir;
            if (string.IsNullOrEmpty(targetDir))
            {
                try
                {
                    using (RegistryKey key = Registry.CurrentUser.OpenSubKey(UNINSTALL_REG_KEY, false))
                    {
                        if (key != null)
                        {
                            targetDir = key.GetValue("InstallLocation") as string;
                        }
                    }
                }
                catch { }
            }
            if (string.IsNullOrEmpty(targetDir))
            {
                targetDir = Path.GetDirectoryName(selfExe);
            }
            if (string.IsNullOrEmpty(targetDir) || !Directory.Exists(targetDir))
            {
                return 0;
            }

            // Safety guard: only wipe a directory that actually looks like a Quantum Studio install
            bool hasMarker =
                File.Exists(Path.Combine(targetDir, "QuantumStudio.exe")) ||
                File.Exists(Path.Combine(targetDir, "run_studio.py")) ||
                File.Exists(Path.Combine(targetDir, "Uninstall.exe"));
            if (!hasMarker)
            {
                if (!silent)
                {
                    MessageBox.Show("未在目标目录检测到 Quantum Studio 安装文件，已取消卸载。", "卸载提示", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
                return 1;
            }

            if (!silent)
            {
                DialogResult dr = MessageBox.Show(
                    "确定要卸载 Quantum Studio v2.3.0 并移除快捷方式与安装目录吗？\n\n目录: " + targetDir,
                    "卸载 Quantum Studio",
                    MessageBoxButtons.YesNo,
                    MessageBoxIcon.Question
                );
                if (dr != DialogResult.Yes) return 0;
            }

            // Remove shortcuts & HKCU registry entry
            try
            {
                string dLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "Quantum Studio.lnk");
                if (File.Exists(dLnk)) File.Delete(dLnk);
            }
            catch { }
            try
            {
                string sLnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Quantum Studio.lnk");
                if (File.Exists(sLnk)) File.Delete(sLnk);
            }
            catch { }
            try
            {
                using (RegistryKey key = Registry.CurrentUser.OpenSubKey(UNINSTALL_REG_KEY, false))
                {
                    string regLoc = key != null ? (key.GetValue("InstallLocation") as string) : null;
                    if (string.IsNullOrEmpty(explicitDir) ||
                        string.Equals(regLoc, targetDir, StringComparison.OrdinalIgnoreCase))
                    {
                        Registry.CurrentUser.DeleteSubKeyTree(UNINSTALL_REG_KEY, false);
                    }
                }
            }
            catch { }

            // Delete subdirectories and all files except running selfExe
            string fullSelf = Path.GetFullPath(selfExe);
            foreach (string subDir in Directory.GetDirectories(targetDir))
            {
                try { Directory.Delete(subDir, true); } catch { }
            }
            foreach (string file in Directory.GetFiles(targetDir))
            {
                if (!string.Equals(Path.GetFullPath(file), fullSelf, StringComparison.OrdinalIgnoreCase))
                {
                    try { File.Delete(file); } catch { }
                }
            }

            // Schedule deferred self-deletion if running from inside targetDir
            if (fullSelf.StartsWith(Path.GetFullPath(targetDir), StringComparison.OrdinalIgnoreCase))
            {
                try
                {
                    string cmdArgs = string.Format(
                        "/C ping 127.0.0.1 -n 2 > nul & del /f /q \"{0}\" & rmdir \"{1}\"",
                        fullSelf,
                        Path.GetFullPath(targetDir)
                    );
                    Process.Start(new ProcessStartInfo
                    {
                        FileName = "cmd.exe",
                        Arguments = cmdArgs,
                        WindowStyle = ProcessWindowStyle.Hidden,
                        CreateNoWindow = true,
                        UseShellExecute = false
                    });
                }
                catch { }
            }
            else
            {
                try { Directory.Delete(targetDir, true); } catch { }
            }

            if (!silent)
            {
                MessageBox.Show("Quantum Studio 已成功卸载。", "卸载完成", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            return 0;
        }

        private static void CreateShortcut(string lnkPath, string targetExe, string workDir, string iconPath)
        {
            try
            {
                Type shellType = Type.GetTypeFromProgID("WScript.Shell");
                if (shellType == null) return;
                object shell = Activator.CreateInstance(shellType);
                object shortcut = shellType.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { lnkPath });
                Type scType = shortcut.GetType();
                scType.InvokeMember("TargetPath", BindingFlags.SetProperty, null, shortcut, new object[] { targetExe });
                scType.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, shortcut, new object[] { workDir });
                scType.InvokeMember("Description", BindingFlags.SetProperty, null, shortcut, new object[] { "Quantum Studio v2 — 量子线路设计与多后端演化工作台" });
                if (File.Exists(iconPath))
                {
                    scType.InvokeMember("IconLocation", BindingFlags.SetProperty, null, shortcut, new object[] { iconPath + ",0" });
                }
                scType.InvokeMember("Save", BindingFlags.InvokeMethod, null, shortcut, null);
            }
            catch
            {
                // Ignore shortcut error in restricted environments
            }
        }
    }

    internal static class Program
    {
        [STAThread]
        private static int Main(string[] args)
        {
            string defaultDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Programs",
                "QuantumStudio"
            );

            bool silent = false;
            bool uninstall = Path.GetFileName(Application.ExecutablePath)
                .Equals("Uninstall.exe", StringComparison.OrdinalIgnoreCase);
            string customDir = null;
            bool noShortcuts = false;

            foreach (string a in args)
            {
                if (a.Equals("/S", StringComparison.OrdinalIgnoreCase) ||
                    a.Equals("--silent", StringComparison.OrdinalIgnoreCase))
                {
                    silent = true;
                }
                else if (a.Equals("--uninstall", StringComparison.OrdinalIgnoreCase) ||
                         a.Equals("/U", StringComparison.OrdinalIgnoreCase))
                {
                    uninstall = true;
                }
                else if (a.Equals("--no-shortcuts", StringComparison.OrdinalIgnoreCase))
                {
                    noShortcuts = true;
                }
                else if (a.StartsWith("/D=", StringComparison.OrdinalIgnoreCase))
                {
                    customDir = a.Substring(3).Trim('"');
                }
            }

            if (uninstall)
            {
                if (!silent)
                {
                    Application.EnableVisualStyles();
                    Application.SetCompatibleTextRenderingDefault(false);
                }
                return InstallerEngine.RunUninstall(customDir, silent);
            }

            if (silent)
            {
                try
                {
                    InstallerEngine.ExtractPayload(
                        string.IsNullOrEmpty(customDir) ? defaultDir : customDir,
                        !noShortcuts,
                        !noShortcuts,
                        null
                    );
                    return 0;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine("Silent install failed: " + ex.Message);
                    return 1;
                }
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SetupForm(string.IsNullOrEmpty(customDir) ? defaultDir : customDir));
            return 0;
        }
    }
}
