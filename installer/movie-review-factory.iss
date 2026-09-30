; Inno Setup script for the Movie Review Factory 1-click Windows installer.
;
; Built by scripts/package_windows.py, which stages the payload (embedded Python +
; ffmpeg/ffprobe + yt-dlp + the app + mrf-launch.vbs) and invokes ISCC with:
;   ISCC /DMyAppVersion=0.2.0 /DPayloadDir=<...\build\win\payload> /DOutputDir=<...\build\win\dist> movie-review-factory.iss
;
; Per-user install (no admin/UAC). The Desktop / Start-menu shortcut launches the
; hidden-window VBScript, which starts the local server and opens the dashboard.

#ifndef MyAppVersion
  #define MyAppVersion "0.2.0"
#endif
#ifndef PayloadDir
  #define PayloadDir "..\build\win\payload"
#endif
#ifndef OutputDir
  #define OutputDir "..\build\win\dist"
#endif

#define MyAppName "Movie Review Factory"
#define MyAppPublisher "Movie Review Factory"
#define MyLauncher "mrf-launch.vbs"

[Setup]
AppId={{7E2B9A54-3C1D-4E77-9B2A-MRF0FACTORY01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\MovieReviewFactory
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Per-user install so no administrator rights are required.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
OutputDir={#OutputDir}
OutputBaseFilename=movie-review-factory-setup-{#MyAppVersion}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "vi"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Tạo lối tắt ngoài Desktop"; GroupDescription: "Lối tắt:"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#MyLauncher}"""; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#MyLauncher}"""; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{sys}\wscript.exe"; Parameters: """{app}\{#MyLauncher}"""; WorkingDir: "{app}"; Description: "Khởi động {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Leave the user's jobs under %LOCALAPPDATA%\MovieReviewFactory in place on uninstall.
Type: filesandordirs; Name: "{app}\python\Lib\site-packages\__pycache__"
