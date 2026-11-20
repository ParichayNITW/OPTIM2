; Pipeline Optima™ — Inno Setup installer script
; Build with: ISCC.exe installer\setup.iss
; Requires Inno Setup 6 (free): https://jrsoftware.org/isinfo.php

#define AppName "Pipeline Optima Pro"
#define AppVersion "1.0.0"
#define AppPublisher "Pipeline Optima"
#define AppURL "https://github.com/ParichayNITW/OPTIM2"
#define AppExeName "PipelineOptima.exe"

[Setup]
AppId={{C9F4B2E1-7D3A-4F9C-8B5E-3A6F8D2C4B17}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
; Installer output — paths relative to this .iss file (inside installer\ folder)
OutputDir=..\dist\installer
OutputBaseFilename=PipelineOptima-Setup-v{#AppVersion}
; Icon is optional — build still succeeds if the .ico file doesn't exist yet
#if FileExists("..\assets\icon.ico")
SetupIconFile=..\assets\icon.ico
#endif
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Require admin rights to install to Program Files
PrivilegesRequired=admin
; Minimum Windows 10
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The entire PyInstaller output directory
Source: "..\dist\pipeline_optima\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcut
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
; Desktop shortcut (optional, user can deselect)
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
; Offer to launch the app after installation
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up the SQLite database on uninstall
Type: files; Name: "{app}\optima.db"
Type: dirifempty; Name: "{app}"

[Code]
// Kill PipelineOptima and all its child processes (Qt WebEngine subprocesses)
// before overwriting files — prevents "DeleteFile failed; code 5" on reinstall.
procedure CurStepChanged(CurStep: TSetupStep);
var ResultCode: Integer;
begin
  if CurStep = ssInstall then begin
    Exec('taskkill.exe', '/F /T /IM PipelineOptima.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Exec('taskkill.exe', '/F /T /IM QtWebEngineProcess.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Sleep(1500);
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := True;
end;
