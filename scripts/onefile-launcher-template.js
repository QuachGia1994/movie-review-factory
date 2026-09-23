/* Movie Review Factory one-file launcher.
 * Generated file: double-click on Windows or run with Node.js.
 */
(function () {
  if (typeof WScript !== "undefined") {
    var shell = new ActiveXObject("WScript.Shell");
    var fso = new ActiveXObject("Scripting.FileSystemObject");
    var self = WScript.ScriptFullName;

    function ensureFolder(folder) {
      if (fso.FolderExists(folder)) {
        return;
      }
      var parent = fso.GetParentFolderName(folder);
      if (parent && !fso.FolderExists(parent)) {
        ensureFolder(parent);
      }
      fso.CreateFolder(folder);
    }

    function findSystemNode() {
      if (shell.ExpandEnvironmentStrings("%MRF_WSH_FORCE_PORTABLE_NODE%") === "1") {
        return "";
      }
      var probe = shell.Exec('cmd.exe /d /s /c "where node 2>nul"');
      return probe.StdOut.ReadAll().replace(/\r/g, "").split("\n")[0];
    }

    function bootstrapNode() {
      var customHome = shell.ExpandEnvironmentStrings("%MRF_HOME%");
      var localApp;
      var bootstrapRoot;
      if (customHome && customHome.indexOf("%MRF_HOME%") < 0) {
        bootstrapRoot = customHome + "\\bootstrap";
      } else {
        localApp = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%");
        if (!localApp || localApp.indexOf("%LOCALAPPDATA%") >= 0) {
          localApp = shell.ExpandEnvironmentStrings("%APPDATA%");
        }
        bootstrapRoot = localApp + "\\MovieReviewFactory\\bootstrap";
      }
      var nodeRoot = bootstrapRoot + "\\node";
      var cached = nodeRoot + "\\node.exe";
      if (fso.FileExists(cached)) {
        return cached;
      }

      ensureFolder(bootstrapRoot);
      var archText = shell.ExpandEnvironmentStrings("%PROCESSOR_ARCHITEW6432%");
      if (!archText || archText.indexOf("%PROCESSOR_ARCHITEW6432%") >= 0) {
        archText = shell.ExpandEnvironmentStrings("%PROCESSOR_ARCHITECTURE%");
      }
      var arch = /ARM64/i.test(archText) ? "arm64" : "x64";
      var ps1 = bootstrapRoot + "\\bootstrap-node-" + (new Date().getTime()) + ".ps1";
      var file = fso.CreateTextFile(ps1, true, true);
      file.WriteLine("$ErrorActionPreference = 'Stop'");
      file.WriteLine("$ProgressPreference = 'SilentlyContinue'");
      file.WriteLine("$target = $args[0]");
      file.WriteLine("$arch = $args[1]");
      file.WriteLine("$index = Invoke-RestMethod -UseBasicParsing 'https://nodejs.org/dist/index.json'");
      file.WriteLine("$needle = 'win-' + $arch + '-zip'");
      file.WriteLine("$entry = $index | Where-Object { $_.lts -and ($_.files -contains $needle) } | Select-Object -First 1");
      file.WriteLine("if (-not $entry) { throw 'No compatible Node LTS portable archive was found.' }");
      file.WriteLine("$version = $entry.version");
      file.WriteLine("$name = 'node-' + $version + '-win-' + $arch + '.zip'");
      file.WriteLine("$base = 'https://nodejs.org/dist/' + $version + '/'");
      file.WriteLine("$zip = Join-Path $env:TEMP ('mrf-' + $name)");
      file.WriteLine("$sums = $zip + '.sha256.txt'");
      file.WriteLine("Invoke-WebRequest -UseBasicParsing -Uri ($base + $name) -OutFile $zip");
      file.WriteLine("Invoke-WebRequest -UseBasicParsing -Uri ($base + 'SHASUMS256.txt') -OutFile $sums");
      file.WriteLine("$line = Get-Content $sums | Where-Object { $_ -match ([regex]::Escape($name) + '$') } | Select-Object -First 1");
      file.WriteLine("if (-not $line) { throw 'Node checksum entry missing.' }");
      file.WriteLine("$expected = ($line -split '\\s+')[0].ToLowerInvariant()");
      file.WriteLine("$actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()");
      file.WriteLine("if ($actual -ne $expected) { throw 'Node archive checksum mismatch.' }");
      file.WriteLine("$tmp = $target + '.tmp'");
      file.WriteLine("Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue");
      file.WriteLine("New-Item -ItemType Directory -Path $tmp -Force | Out-Null");
      file.WriteLine("Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force");
      file.WriteLine("$inner = Get-ChildItem -LiteralPath $tmp -Directory | Select-Object -First 1");
      file.WriteLine("if (-not $inner) { throw 'Node archive layout is invalid.' }");
      file.WriteLine("Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue");
      file.WriteLine("Move-Item -LiteralPath $inner.FullName -Destination $target");
      file.WriteLine("Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue");
      file.WriteLine("Remove-Item -LiteralPath $zip,$sums -Force -ErrorAction SilentlyContinue");
      file.Close();

      var psCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' +
        ps1.replace(/"/g, '""') + '" "' + nodeRoot.replace(/"/g, '""') + '" "' + arch + '"';
      var exitCode = shell.Run(psCommand, 1, true);
      try {
        fso.DeleteFile(ps1, true);
      } catch (ignore) {}
      if (exitCode !== 0 || !fso.FileExists(cached)) {
        shell.Popup(
          "Không thể tự cài Node.js portable. Kiểm tra kết nối Internet rồi mở lại file.",
          0,
          "Movie Review Factory",
          16
        );
        WScript.Quit(2);
      }
      return cached;
    }

    var found = findSystemNode();
    if (!found) {
      shell.Popup(
        "Lần chạy đầu: đang chuẩn bị Node.js portable. Quá trình này chỉ thực hiện một lần.",
        5,
        "Movie Review Factory",
        64
      );
      found = bootstrapNode();
    }

    var probeOnly = false;
    var forwarded = "";
    var i;
    for (i = 0; i < WScript.Arguments.length; i += 1) {
      var value = String(WScript.Arguments.Item(i));
      if (value === "--wsh-self-test") {
        probeOnly = true;
      } else {
        forwarded += ' "' + value.replace(/"/g, '""') + '"';
      }
    }
    if (probeOnly) {
      WScript.Echo("MRF_WSH_OK " + found);
      WScript.Quit(0);
    }

    var command = '"' + found.replace(/"/g, '""') + '" "' +
      self.replace(/"/g, '""') + '" --node' + forwarded;
    try {
      shell.Run(command, 1, false);
    } catch (error) {
      shell.Popup(
        "Không thể khởi động Node.js: " + error.message,
        0,
        "Movie Review Factory",
        16
      );
      WScript.Quit(3);
    }
    WScript.Quit(0);
  }
})();

if (typeof require !== "function") {
  throw new Error("Movie Review Factory cần Node.js.");
}

var fs = require("fs");
var path = require("path");
var os = require("os");
var crypto = require("crypto");
var childProcess = require("child_process");

var APP_NAME = "MovieReviewFactory";
var VERSION = __MRF_VERSION__;
var BUNDLE_HASH = __MRF_HASH__;
var BUNDLE = __MRF_BUNDLE__;
var MANAGED_PYTHON = "3.12";
var CORE_REQUIREMENTS = [
  "pydantic>=2.11,<3",
  "typer>=0.16,<1"
];
var FULL_REQUIREMENTS = CORE_REQUIREMENTS.concat([
  "faster-whisper>=1.1,<2",
  "edge-tts>=7,<8"
]);
var UV_RELEASE_API = "https://api.github.com/repos/astral-sh/uv/releases/latest";
var FFMPEG_RELEASE_API = "https://api.github.com/repos/GyanD/codexffmpeg/releases/latest";
var FFMPEG_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip";
var FFMPEG_SHA_URL = FFMPEG_ZIP_URL + ".sha256";

function log(message) {
  process.stdout.write("[MRF] " + message + "\n");
}

function fail(message, code) {
  process.stderr.write("[MRF] LỖI: " + message + "\n");
  process.exit(typeof code === "number" ? code : 1);
}

function hasArg(name) {
  return process.argv.indexOf(name) !== -1;
}

function argValue(name) {
  var index = process.argv.indexOf(name);
  if (index >= 0 && index + 1 < process.argv.length) {
    return process.argv[index + 1];
  }
  return null;
}

function sha12(value) {
  return crypto.createHash("sha256").update(String(value)).digest("hex").slice(0, 12);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function writableDir(preferred, fallback) {
  try {
    ensureDir(preferred);
    var probe = path.join(preferred, ".mrf-write-" + process.pid);
    fs.writeFileSync(probe, "ok");
    fs.unlinkSync(probe);
    return preferred;
  } catch (error) {
    ensureDir(fallback);
    return fallback;
  }
}

function appDataRoot() {
  var requested = argValue("--home");
  if (requested) {
    return path.resolve(requested);
  }
  if (process.env.MRF_HOME) {
    return path.resolve(process.env.MRF_HOME);
  }
  var base = process.env.LOCALAPPDATA || process.env.APPDATA ||
    path.join(os.homedir(), ".movie-review-factory");
  return path.join(base, APP_NAME);
}

function runtimeRoot() {
  return path.join(appDataRoot(), "runtime", BUNDLE_HASH.slice(0, 16));
}

function ensureRuntime() {
  var root = runtimeRoot();
  var marker = path.join(root, ".bundle.json");
  try {
    var info = JSON.parse(fs.readFileSync(marker, "utf8"));
    if (info.hash === BUNDLE_HASH && info.version === VERSION) {
      return root;
    }
  } catch (error) {}

  var parent = path.dirname(root);
  ensureDir(parent);
  var temp = root + ".tmp-" + process.pid;
  fs.rmSync(temp, { recursive: true, force: true });
  ensureDir(temp);

  Object.keys(BUNDLE).forEach(function (relative) {
    var target = path.join(temp, relative.replace(/\//g, path.sep));
    ensureDir(path.dirname(target));
    fs.writeFileSync(target, Buffer.from(BUNDLE[relative], "base64"));
  });
  fs.writeFileSync(
    path.join(temp, ".bundle.json"),
    JSON.stringify({ hash: BUNDLE_HASH, version: VERSION }, null, 2),
    "utf8"
  );

  fs.rmSync(root, { recursive: true, force: true });
  fs.renameSync(temp, root);
  return root;
}

function runSync(command, args, options) {
  var opts = options || {};
  opts.encoding = "utf8";
  opts.windowsHide = true;
  if (!opts.timeout) {
    opts.timeout = 10000;
  }
  return childProcess.spawnSync(command, args, opts);
}

function toolchainRoot() {
  return path.join(appDataRoot(), "toolchain");
}

function bootstrapRoot() {
  return path.join(appDataRoot(), "bootstrap");
}

function noNetwork() {
  return process.env.MRF_NO_NETWORK === "1" || hasArg("--no-network");
}

function runtimeProfile() {
  var explicit = process.env.MRF_RUNTIME_PROFILE || argValue("--runtime-profile");
  return String(explicit || "full").toLowerCase() === "core" ? "core" : "full";
}

function mergeEnv(extra) {
  var env = {};
  Object.keys(process.env).forEach(function (key) {
    env[key] = process.env[key];
  });
  Object.keys(extra || {}).forEach(function (key) {
    env[key] = extra[key];
  });
  return env;
}

function prependEnvPath(env, directory) {
  var current = "";
  Object.keys(env).forEach(function (key) {
    if (key.toUpperCase() === "PATH") {
      if (!current) {
        current = String(env[key] || "");
      }
      delete env[key];
    }
  });
  env[process.platform === "win32" ? "Path" : "PATH"] =
    directory + path.delimiter + current;
  return env;
}

function runPowerShellScript(script, args, timeout) {
  if (process.platform !== "win32") {
    return { status: 1, stdout: "", stderr: "PowerShell bootstrap requires Windows." };
  }
  ensureDir(bootstrapRoot());
  var ps1 = path.join(
    bootstrapRoot(),
    "mrf-bootstrap-" + process.pid + "-" + Date.now() + ".ps1"
  );
  fs.writeFileSync(ps1, script, "utf8");
  try {
    return runSync(
      "powershell.exe",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1].concat(args || []),
      { timeout: timeout || 600000 }
    );
  } finally {
    try {
      fs.unlinkSync(ps1);
    } catch (error) {}
  }
}

function firstExisting(paths) {
  for (var i = 0; i < paths.length; i += 1) {
    if (paths[i] && fs.existsSync(paths[i])) {
      return paths[i];
    }
  }
  return null;
}

function findFileRecursive(root, name) {
  if (!fs.existsSync(root)) {
    return null;
  }
  var stack = [root];
  while (stack.length) {
    var current = stack.pop();
    var entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch (error) {
      continue;
    }
    for (var i = 0; i < entries.length; i += 1) {
      var entry = entries[i];
      var target = path.join(current, entry.name);
      if (entry.isFile() && entry.name.toLowerCase() === name.toLowerCase()) {
        return target;
      }
      if (entry.isDirectory()) {
        stack.push(target);
      }
    }
  }
  return null;
}

function sha256File(file) {
  var hash = crypto.createHash("sha256");
  var fd = fs.openSync(file, "r");
  var buffer = Buffer.alloc(1024 * 1024);
  try {
    while (true) {
      var count = fs.readSync(fd, buffer, 0, buffer.length, null);
      if (!count) {
        break;
      }
      hash.update(buffer.subarray(0, count));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest("hex");
}

function downloadFile(url, target, timeout) {
  ensureDir(path.dirname(target));
  var curl = commandPath("curl.exe") || commandPath("curl");
  if (curl) {
    var maxSeconds = Math.max(30, Math.ceil((timeout || 900000) / 1000));
    var hadPartial = fs.existsSync(target) && fs.statSync(target).size > 0;
    var curlArgs = [
      "-L", "--fail", "--show-error",
      "--retry", "3", "--retry-all-errors", "--retry-delay", "2",
      "--connect-timeout", "30", "--max-time", String(maxSeconds),
      "--progress-bar"
    ];
    if (hadPartial) {
      curlArgs.push("--continue-at", "-");
    }
    curlArgs.push("--output", target, url);

    var result = childProcess.spawnSync(
      curl,
      curlArgs,
      {
        windowsHide: false,
        stdio: "inherit",
        timeout: (timeout || 900000) + 5000
      }
    );
    if (result.status === 0 && fs.existsSync(target) && fs.statSync(target).size > 0) {
      return;
    }

    if (hadPartial) {
      try {
        fs.unlinkSync(target);
      } catch (error) {}
      result = childProcess.spawnSync(
        curl,
        [
          "-L", "--fail", "--show-error",
          "--retry", "3", "--retry-all-errors", "--retry-delay", "2",
          "--connect-timeout", "30", "--max-time", String(maxSeconds),
          "--progress-bar", "--output", target, url
        ],
        {
          windowsHide: false,
          stdio: "inherit",
          timeout: (timeout || 900000) + 5000
        }
      );
      if (result.status === 0 && fs.existsSync(target) && fs.statSync(target).size > 0) {
        return;
      }
    }

    if (result.error) {
      throw result.error;
    }
    throw new Error("curl download failed: " + url);
  }

  var script = [
    "$ErrorActionPreference = 'Stop'",
    "$ProgressPreference = 'SilentlyContinue'",
    "Invoke-WebRequest -UseBasicParsing -Uri $args[0] -OutFile $args[1]"
  ].join("\n");
  var fallback = runPowerShellScript(script, [url, target], timeout || 900000);
  if (
    fallback.status !== 0 ||
    !fs.existsSync(target) ||
    fs.statSync(target).size <= 0
  ) {
    throw new Error(
      "PowerShell download failed: " +
      String(fallback.stderr || fallback.stdout || fallback.error || "")
    );
  }
}

function pythonCandidates() {
  var list = [];
  if (process.env.MRF_PYTHON) {
    list.push({ command: process.env.MRF_PYTHON, prefix: [] });
  }
  if (process.platform === "win32") {
    list.push({ command: "py", prefix: ["-3"] });
  }
  list.push({ command: "python", prefix: [] });
  list.push({ command: "python3", prefix: [] });
  return list;
}

function detectPython() {
  var candidates = pythonCandidates();
  var i;
  for (i = 0; i < candidates.length; i += 1) {
    var candidate = candidates[i];
    var args = candidate.prefix.concat([
      "-c",
      "import json,sys; print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor,'exe':sys.executable}))"
    ]);
    var result = runSync(candidate.command, args);
    if (result.status === 0) {
      try {
        var info = JSON.parse(String(result.stdout).trim());
        if (info.major > 3 || (info.major === 3 && info.minor >= 11)) {
          candidate.info = info;
          return candidate;
        }
      } catch (error) {}
    }
  }
  return null;
}

function pythonArgs(py, args) {
  return py.prefix.concat(args);
}

function checkCoreDependencies(py) {
  var result = runSync(
    py.command,
    pythonArgs(py, [
      "-c",
      "import pydantic, typer; from pydantic import BaseModel; print(pydantic.__version__)"
    ])
  );
  return result.status === 0;
}

function checkRuntimeDependencies(py, env) {
  var imports = runtimeProfile() === "core"
    ? "import pydantic, typer"
    : "import pydantic, typer, faster_whisper, edge_tts";
  var result = runSync(
    py.command,
    pythonArgs(py, ["-c", imports]),
    { env: env || process.env, timeout: 15000 }
  );
  return result.status === 0;
}

function ensureCoreDependencies(py) {
  if (checkCoreDependencies(py)) {
    return true;
  }
  if (hasArg("--no-auto-install") || process.env.MRF_NO_AUTO_INSTALL === "1") {
    return false;
  }
  log("Thiếu dependency lõi; đang cài pydantic + typer một lần...");
  var install = childProcess.spawnSync(
    py.command,
    pythonArgs(py, [
      "-m", "pip", "install", "--user",
      "pydantic>=2.11,<3", "typer>=0.16,<1"
    ]),
    { stdio: "inherit", windowsHide: false }
  );
  return install.status === 0 && checkCoreDependencies(py);
}

function ensureSystemRuntimeDependencies(py) {
  if (checkRuntimeDependencies(py, process.env)) {
    return true;
  }
  if (hasArg("--no-auto-install") || process.env.MRF_NO_AUTO_INSTALL === "1" || noNetwork()) {
    return false;
  }
  var requirements = requirementsForProfile();
  log("Đang chuẩn bị dependency cho Python hệ thống...");
  var install = childProcess.spawnSync(
    py.command,
    pythonArgs(py, ["-m", "pip", "install", "--user"].concat(requirements)),
    { stdio: "inherit", windowsHide: false }
  );
  return install.status === 0 && checkRuntimeDependencies(py, process.env);
}

function commandPath(command) {
  var probe;
  if (process.platform === "win32") {
    probe = runSync("where", [command], { timeout: 4000 });
  } else {
    probe = runSync("which", [command], { timeout: 4000 });
  }
  if (probe.status !== 0) {
    return null;
  }
  var lines = String(probe.stdout || "").replace(/\r/g, "").split("\n");
  return lines[0] ? lines[0].trim() : null;
}

function commandExists(command) {
  return Boolean(commandPath(command));
}

function ensureUv() {
  var configured = process.env.MRF_UV;
  if (configured && fs.existsSync(configured)) {
    return configured;
  }
  var targetDir = path.join(toolchainRoot(), "uv");
  var target = path.join(targetDir, process.platform === "win32" ? "uv.exe" : "uv");
  if (fs.existsSync(target)) {
    return target;
  }
  if (process.platform !== "win32") {
    var systemUv = commandPath("uv");
    if (systemUv) {
      return systemUv;
    }
    fail("Zero-install Python bootstrap hiện chỉ hỗ trợ Windows.");
  }
  if (noNetwork()) {
    fail("Thiếu uv portable trong cache và chế độ --no-network đang bật.");
  }

  log("Lần chạy đầu: đang chuẩn bị Python runtime manager (uv)...");
  ensureDir(targetDir);
  var arch = process.arch === "arm64" ? "aarch64" : "x86_64";
  var asset = "uv-" + arch + "-pc-windows-msvc.zip";
  var script = [
    "$ErrorActionPreference = 'Stop'",
    "$ProgressPreference = 'SilentlyContinue'",
    "$target = $args[0]",
    "$assetName = $args[1]",
    "$api = $args[2]",
    "$headers = @{ 'User-Agent' = 'MovieReviewFactory' }",
    "$release = Invoke-RestMethod -UseBasicParsing -Headers $headers -Uri $api",
    "$asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1",
    "if (-not $asset) { throw ('uv asset missing: ' + $assetName) }",
    "$checksumAsset = $release.assets | Where-Object { $_.name -eq ($assetName + '.sha256') } | Select-Object -First 1",
    "$zip = Join-Path $env:TEMP ('mrf-' + $assetName)",
    "$checksum = $zip + '.sha256'",
    "Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri $asset.browser_download_url -OutFile $zip",
    "$expected = $null",
    "if ($asset.digest -and $asset.digest.StartsWith('sha256:')) {",
    "  $expected = $asset.digest.Substring(7).ToLowerInvariant()",
    "} elseif ($checksumAsset) {",
    "  Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri $checksumAsset.browser_download_url -OutFile $checksum",
    "  $expected = ((Get-Content -LiteralPath $checksum | Select-Object -First 1) -split '\\s+')[0].Trim().ToLowerInvariant()",
    "} else {",
    "  throw 'uv release does not expose a SHA256 digest.'",
    "}",
    "$actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()",
    "if (-not $expected -or $actual -ne $expected) { throw 'uv archive checksum mismatch.' }",
    "$tmp = (Split-Path -Parent $target) + '.tmp'",
    "Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue",
    "New-Item -ItemType Directory -Path $tmp -Force | Out-Null",
    "Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force",
    "$exe = Get-ChildItem -LiteralPath $tmp -Recurse -Filter uv.exe | Select-Object -First 1",
    "if (-not $exe) { throw 'uv.exe missing from release archive.' }",
    "New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null",
    "Copy-Item -LiteralPath $exe.FullName -Destination $target -Force",
    "Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue",
    "Remove-Item -LiteralPath $zip,$checksum -Force -ErrorAction SilentlyContinue"
  ].join("\n");
  var result = runPowerShellScript(script, [target, asset, UV_RELEASE_API], 180000);
  if (result.status !== 0 || !fs.existsSync(target)) {
    fail("Không thể tải uv portable: " + String(result.stderr || result.stdout || "").trim());
  }
  return target;
}

function managedEnvironment() {
  return mergeEnv({
    UV_PYTHON_INSTALL_DIR: path.join(toolchainRoot(), "python"),
    UV_CACHE_DIR: path.join(appDataRoot(), "cache", "uv"),
    UV_NO_PROGRESS: "1",
    HF_HOME: path.join(appDataRoot(), "cache", "huggingface")
  });
}

function pythonInfo(py, env) {
  var result = runSync(
    py.command,
    pythonArgs(py, [
      "-c",
      "import json,sys; print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor,'exe':sys.executable}))"
    ]),
    { env: env || process.env, timeout: 10000 }
  );
  if (result.status !== 0) {
    return null;
  }
  try {
    return JSON.parse(String(result.stdout).trim());
  } catch (error) {
    return null;
  }
}

function requirementsForProfile() {
  return runtimeProfile() === "core" ? CORE_REQUIREMENTS : FULL_REQUIREMENTS;
}

function ensureManagedPython() {
  var uv = ensureUv();
  var env = managedEnvironment();
  var venv = path.join(toolchainRoot(), "venv-py" + MANAGED_PYTHON.replace(".", ""));
  var pythonExe = path.join(venv, "Scripts", "python.exe");
  if (!fs.existsSync(pythonExe)) {
    if (noNetwork()) {
      fail("Thiếu Python managed trong cache và chế độ --no-network đang bật.");
    }
    log("Lần chạy đầu: đang chuẩn bị Python " + MANAGED_PYTHON + " portable...");
    ensureDir(toolchainRoot());
    var create = childProcess.spawnSync(
      uv,
      ["venv", "--python", MANAGED_PYTHON, venv],
      { env: env, stdio: "inherit", windowsHide: false }
    );
    if (create.status !== 0 || !fs.existsSync(pythonExe)) {
      fail("Không thể tạo Python runtime bằng uv.");
    }
  }

  var requirements = requirementsForProfile();
  var requirementsHash = crypto.createHash("sha256")
    .update(JSON.stringify(requirements))
    .digest("hex");
  var marker = path.join(venv, ".mrf-requirements.json");
  var ready = false;
  try {
    var markerInfo = JSON.parse(fs.readFileSync(marker, "utf8"));
    ready = markerInfo.hash === requirementsHash && checkRuntimeDependencies(
      { command: pythonExe, prefix: [] },
      env
    );
  } catch (error) {}

  if (!ready) {
    if (noNetwork()) {
      fail("Python runtime chưa đủ dependency và chế độ --no-network đang bật.");
    }
    log(
      runtimeProfile() === "core"
        ? "Đang cài dependency lõi..."
        : "Lần chạy đầu: đang cài Whisper + TTS + dependency ứng dụng..."
    );
    var install = childProcess.spawnSync(
      uv,
      ["pip", "install", "--python", pythonExe].concat(requirements),
      { env: env, stdio: "inherit", windowsHide: false }
    );
    if (install.status !== 0) {
      fail("Không thể cài Python dependency vào runtime managed.");
    }
    if (!checkRuntimeDependencies({ command: pythonExe, prefix: [] }, env)) {
      fail("Python dependency đã cài nhưng import verification thất bại.");
    }
    fs.writeFileSync(
      marker,
      JSON.stringify({ hash: requirementsHash, requirements: requirements }, null, 2),
      "utf8"
    );
  }

  var py = { command: pythonExe, prefix: [] };
  py.info = pythonInfo(py, env);
  if (!py.info) {
    fail("Python managed đã tạo nhưng không thể khởi động.");
  }
  return { python: py, env: env, uv: uv };
}

function validateFfmpegBin(binDir) {
  var ffmpeg = path.join(binDir, process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg");
  var ffprobe = path.join(binDir, process.platform === "win32" ? "ffprobe.exe" : "ffprobe");
  if (!fs.existsSync(ffmpeg) || !fs.existsSync(ffprobe)) {
    return false;
  }
  var ffmpegCheck = runSync(ffmpeg, ["-version"], { timeout: 10000 });
  var ffprobeCheck = runSync(ffprobe, ["-version"], { timeout: 10000 });
  return ffmpegCheck.status === 0 && ffprobeCheck.status === 0;
}

function ffmpegReleaseMeta() {
  var script = [
    "$ErrorActionPreference = 'Stop'",
    "$headers = @{ 'User-Agent' = 'MovieReviewFactory' }",
    "$release = Invoke-RestMethod -UseBasicParsing -Headers $headers -Uri $args[0]",
    "$asset = $release.assets | Where-Object { $_.name -match '-essentials_build\\.zip$' } | Select-Object -First 1",
    "if (-not $asset) { throw 'FFmpeg essentials ZIP asset missing from latest release.' }",
    "if (-not $asset.digest -or -not $asset.digest.StartsWith('sha256:')) { throw 'FFmpeg release asset digest missing.' }",
    "[pscustomobject]@{ url = $asset.browser_download_url; digest = $asset.digest.Substring(7); name = $asset.name } | ConvertTo-Json -Compress"
  ].join("\n");
  var result = runPowerShellScript(script, [FFMPEG_RELEASE_API], 60000);
  if (result.status !== 0) {
    return null;
  }
  try {
    return JSON.parse(String(result.stdout || "").trim());
  } catch (error) {
    return null;
  }
}

function ensureFfmpeg() {
  var configured = process.env.MRF_FFMPEG_BIN;
  if (configured) {
    var configuredFfmpeg = path.join(configured, "ffmpeg.exe");
    var configuredFfprobe = path.join(configured, "ffprobe.exe");
    if (
      fs.existsSync(configuredFfmpeg) &&
      fs.existsSync(configuredFfprobe) &&
      validateFfmpegBin(configured)
    ) {
      return configured;
    }
  }

  var systemFfmpeg = commandPath("ffmpeg");
  var systemFfprobe = commandPath("ffprobe");
  if (
    systemFfmpeg && systemFfprobe &&
    process.env.MRF_FORCE_PORTABLE_FFMPEG !== "1" &&
    !hasArg("--portable-ffmpeg")
  ) {
    return path.dirname(systemFfmpeg);
  }

  var ffmpegRoot = path.join(toolchainRoot(), "ffmpeg");
  var cachedFfmpeg = findFileRecursive(ffmpegRoot, "ffmpeg.exe");
  var cachedFfprobe = findFileRecursive(ffmpegRoot, "ffprobe.exe");
  if (
    cachedFfmpeg &&
    cachedFfprobe &&
    validateFfmpegBin(path.dirname(cachedFfmpeg))
  ) {
    return path.dirname(cachedFfmpeg);
  }
  if (process.platform !== "win32") {
    fail("Không tìm thấy FFmpeg/FFprobe trên PATH.");
  }
  if (noNetwork()) {
    fail("Thiếu FFmpeg portable trong cache và chế độ --no-network đang bật.");
  }

  log("Lần chạy đầu: đang chuẩn bị FFmpeg portable...");
  ensureDir(toolchainRoot());
  var zip = path.join(os.tmpdir(), "mrf-ffmpeg-release-essentials.zip");
  var sha = zip + ".sha256";
  try {
    var release = ffmpegReleaseMeta();
    var expected;
    if (release && release.url && release.digest) {
      log("Đang tải FFmpeg essentials từ GitHub mirror...");
      downloadFile(release.url, zip, 900000);
      expected = String(release.digest).trim().toLowerCase();
    } else {
      log("GitHub mirror metadata không khả dụng; dùng Gyan direct...");
      downloadFile(FFMPEG_ZIP_URL, zip, 900000);
      downloadFile(FFMPEG_SHA_URL, sha, 60000);
      expected = String(fs.readFileSync(sha, "utf8")).trim().split(/\s+/)[0].toLowerCase();
    }
    var actual = sha256File(zip).toLowerCase();
    if (!expected || actual !== expected) {
      fail("FFmpeg archive checksum mismatch.");
    }

    var script = [
      "$ErrorActionPreference = 'Stop'",
      "$target = $args[0]",
      "$zip = $args[1]",
      "$tmp = $target + '.tmp'",
      "Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue",
      "New-Item -ItemType Directory -Path $tmp -Force | Out-Null",
      "Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force",
      "$inner = Get-ChildItem -LiteralPath $tmp -Directory | Select-Object -First 1",
      "if (-not $inner) { throw 'FFmpeg archive layout is invalid.' }",
      "Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue",
      "Move-Item -LiteralPath $inner.FullName -Destination $target",
      "Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue"
    ].join("\n");
    var result = runPowerShellScript(script, [ffmpegRoot, zip], 600000);
    if (result.status !== 0) {
      fail(
        "Không thể giải nén FFmpeg portable: " +
        String(result.stderr || result.stdout || result.error || "").trim()
      );
    }
  } catch (error) {
    fail("Không thể tải FFmpeg portable: " + error.message);
  } finally {
    try {
      fs.unlinkSync(zip);
    } catch (ignoreZip) {}
    try {
      fs.unlinkSync(sha);
    } catch (ignoreSha) {}
  }
  cachedFfmpeg = findFileRecursive(ffmpegRoot, "ffmpeg.exe");
  cachedFfprobe = findFileRecursive(ffmpegRoot, "ffprobe.exe");
  if (!cachedFfmpeg || !cachedFfprobe) {
    fail("FFmpeg portable đã tải nhưng thiếu ffmpeg.exe/ffprobe.exe.");
  }
  var portableBin = path.dirname(cachedFfmpeg);
  if (!validateFfmpegBin(portableBin)) {
    fail("FFmpeg portable đã tải nhưng không thể thực thi.");
  }
  return portableBin;
}

function ensureToolchain() {
  var useSystem = process.env.MRF_USE_SYSTEM_RUNTIME === "1" || hasArg("--system-runtime");
  var pythonRuntime;
  if (useSystem || process.platform !== "win32") {
    var py = detectPython();
    if (!py) {
      fail("Không tìm thấy Python >= 3.11.");
    }
    if (!ensureSystemRuntimeDependencies(py)) {
      fail("Không thể chuẩn bị dependency Python cho runtime hệ thống.");
    }
    py.info = py.info || pythonInfo(py, process.env);
    pythonRuntime = { python: py, env: mergeEnv({}), uv: null };
  } else {
    pythonRuntime = ensureManagedPython();
  }

  var ffmpegBin = ensureFfmpeg();
  var childEnv = mergeEnv(pythonRuntime.env || {});
  prependEnvPath(childEnv, ffmpegBin);
  childEnv.MRF_FFMPEG_BIN = ffmpegBin;
  childEnv.MRF_WHISPER_CACHE = path.join(appDataRoot(), "models", "whisper");
  if (noNetwork()) {
    childEnv.MRF_WHISPER_OFFLINE = "1";
    childEnv.HF_HUB_OFFLINE = "1";
  }
  return {
    python: pythonRuntime.python,
    env: childEnv,
    uv: pythonRuntime.uv,
    ffmpegBin: ffmpegBin,
    profile: runtimeProfile(),
    managed: !useSystem && process.platform === "win32"
  };
}

function runtimeWarnings() {
  var warnings = [];
  if (!commandExists("claude") && !process.env.MRF_CLAUDE_BIN) {
    warnings.push("Claude Code chưa có: content_agent=scaffold vẫn dùng được đầy đủ media.");
  }
  return warnings;
}

function lockPath(scriptDir) {
  return path.join(appDataRoot(), "server-" + sha12(scriptDir.toLowerCase()) + ".json");
}

function pidAlive(pid) {
  if (!pid || typeof pid !== "number") {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return false;
  }
}

function readLock(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    return null;
  }
}

function openBrowser(url) {
  if (hasArg("--no-browser")) {
    return;
  }
  var opener;
  if (process.platform === "win32") {
    opener = childProcess.spawn(
      "cmd.exe",
      ["/d", "/s", "/c", "start", "", url],
      { detached: true, stdio: "ignore", windowsHide: true }
    );
  } else if (process.platform === "darwin") {
    opener = childProcess.spawn("open", [url], { detached: true, stdio: "ignore" });
  } else {
    opener = childProcess.spawn("xdg-open", [url], { detached: true, stdio: "ignore" });
  }
  opener.unref();
}

function chooseJobsRoot(scriptDir) {
  var requested = argValue("--jobs-root");
  if (requested) {
    return path.resolve(requested);
  }
  var preferred = path.join(scriptDir, "jobs");
  var fallback = path.join(appDataRoot(), "jobs");
  return writableDir(preferred, fallback);
}

function pythonBootstrapCode() {
  return [
    "import json, pathlib, sys",
    "runtime = pathlib.Path(sys.argv[1])",
    "jobs = pathlib.Path(sys.argv[2])",
    "port = int(sys.argv[3])",
    "sys.path.insert(0, str(runtime))",
    "from movie_review_factory.webapp import create_server",
    "server = create_server('127.0.0.1', port, jobs)",
    "host, bound_port = server.server_address[:2]",
    "print('MRF_READY ' + json.dumps({'url': f'http://{host}:{bound_port}', 'jobs': str(jobs)}), flush=True)",
    "try:",
    "    server.serve_forever()",
    "except KeyboardInterrupt:",
    "    pass",
    "finally:",
    "    server.server_close()"
  ].join("\n");
}

function pythonSelfTestCode() {
  return [
    "import json, pathlib, sys, threading, urllib.request",
    "runtime = pathlib.Path(sys.argv[1])",
    "jobs = pathlib.Path(sys.argv[2])",
    "sys.path.insert(0, str(runtime))",
    "from movie_review_factory.webapp import create_server",
    "server = create_server('127.0.0.1', 0, jobs)",
    "host, port = server.server_address[:2]",
    "thread = threading.Thread(target=server.serve_forever, daemon=True)",
    "thread.start()",
    "base = f'http://{host}:{port}'",
    "try:",
    "    with urllib.request.urlopen(base + '/', timeout=5) as response:",
    "        html = response.read()",
    "        root_ok = response.status == 200 and b'<title>' in html and b'jobList' in html",
    "    with urllib.request.urlopen(base + '/api/jobs', timeout=5) as response:",
    "        payload = json.loads(response.read().decode('utf-8'))",
    "        api_ok = response.status == 200 and isinstance(payload.get('jobs'), list)",
    "finally:",
    "    server.shutdown()",
    "    server.server_close()",
    "    thread.join(timeout=5)",
    "ok = bool(root_ok and api_ok)",
    "print(json.dumps({'ok': ok, 'url': base, 'jobs': str(jobs), 'root_ok': root_ok, 'api_ok': api_ok}))",
    "raise SystemExit(0 if ok else 1)"
  ].join("\n");
}

function selfTest(py, runtime, jobs, env) {
  var result = runSync(
    py.command,
    pythonArgs(py, ["-u", "-c", pythonSelfTestCode(), runtime, jobs]),
    { timeout: 15000, env: env }
  );
  if (result.status !== 0) {
    process.stderr.write(String(result.stderr || result.stdout || ""));
    if (result.error) {
      process.stderr.write("\n[MRF] SELF-TEST spawn error: " + result.error.message + "\n");
    } else {
      process.stderr.write(
        "\n[MRF] SELF-TEST failed: status=" + result.status +
        " signal=" + (result.signal || "") + "\n"
      );
    }
    return false;
  }
  log("SELF-TEST " + String(result.stdout).trim());
  return true;
}

function startServer(py, runtime, jobs, scriptDir, env) {
  ensureDir(jobs);
  ensureDir(appDataRoot());
  var lockFile = lockPath(scriptDir);
  var existing = readLock(lockFile);
  if (existing && pidAlive(existing.pid) && existing.url) {
    log("Server đã chạy: " + existing.url);
    openBrowser(existing.url);
    return;
  }
  try {
    fs.unlinkSync(lockFile);
  } catch (error) {}

  var requestedPort = argValue("--port");
  var port = requestedPort ? parseInt(requestedPort, 10) : 0;
  if (!isFinite(port) || port < 0 || port > 65535) {
    fail("--port không hợp lệ.");
  }

  log("Python: " + py.info.exe + " (" + py.info.major + "." + py.info.minor + ")");
  log("Runtime cache: " + runtime);
  log("Jobs: " + jobs);
  runtimeWarnings().forEach(function (warning) {
    log("CẢNH BÁO: " + warning);
  });

  var child = childProcess.spawn(
    py.command,
    pythonArgs(py, ["-u", "-c", pythonBootstrapCode(), runtime, jobs, String(port)]),
    {
      cwd: scriptDir,
      env: env,
      windowsHide: false,
      stdio: ["ignore", "pipe", "pipe"]
    }
  );

  var ready = false;
  var stdoutBuffer = "";

  function removeOwnLock() {
    var current = readLock(lockFile);
    if (current && current.pid === process.pid) {
      try {
        fs.unlinkSync(lockFile);
      } catch (error) {}
    }
  }

  function shutdown() {
    removeOwnLock();
    if (child && !child.killed) {
      try {
        child.kill();
      } catch (error) {}
    }
  }

  process.on("SIGINT", function () {
    shutdown();
    process.exit(0);
  });
  process.on("SIGTERM", function () {
    shutdown();
    process.exit(0);
  });
  process.on("exit", removeOwnLock);

  child.stdout.on("data", function (chunk) {
    stdoutBuffer += String(chunk);
    var lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop();
    lines.forEach(function (line) {
      if (line.indexOf("MRF_READY ") === 0) {
        try {
          var readyInfo = JSON.parse(line.slice("MRF_READY ".length));
          fs.writeFileSync(
            lockFile,
            JSON.stringify({
              pid: process.pid,
              child_pid: child.pid,
              url: readyInfo.url,
              jobs: readyInfo.jobs,
              script: __filename,
              version: VERSION,
              hash: BUNDLE_HASH
            }, null, 2),
            "utf8"
          );
          ready = true;
          log("Local web: " + readyInfo.url);
          openBrowser(readyInfo.url);
        } catch (error) {
          fail("Không đọc được trạng thái server: " + error.message);
        }
      } else if (line) {
        process.stdout.write(line + "\n");
      }
    });
  });

  child.stderr.on("data", function (chunk) {
    process.stderr.write(String(chunk));
  });

  child.on("exit", function (code) {
    removeOwnLock();
    if (!ready) {
      fail("Server dừng trước khi sẵn sàng (exit " + code + ").", code || 1);
    }
    log("Server đã dừng.");
  });
}

function main() {
  var scriptDir = path.dirname(path.resolve(__filename));
  var jobs = chooseJobsRoot(scriptDir);
  var runtime = ensureRuntime();

  if (hasArg("--extract-only")) {
    log("Đã giải nén runtime: " + runtime);
    return;
  }

  var toolchain = ensureToolchain();
  log(
    "Python: " + toolchain.python.info.exe +
    " (" + toolchain.python.info.major + "." + toolchain.python.info.minor + ")" +
    (toolchain.managed ? " [managed]" : " [system]")
  );
  log("FFmpeg: " + toolchain.ffmpegBin);
  log("Runtime profile: " + toolchain.profile);

  if (hasArg("--self-test")) {
    process.exit(
      selfTest(toolchain.python, runtime, jobs, toolchain.env) ? 0 : 1
    );
  }
  startServer(toolchain.python, runtime, jobs, scriptDir, toolchain.env);
}

main();
