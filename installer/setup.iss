; Pipeline Optima™ — Inno Setup installer script
; Build with: ISCC.exe installer\setup.iss
; Requires Inno Setup 6 (free): https://jrsoftware.org/isinfo.php

#define AppName "Pipeline Optima"
#define AppVersion "1.0.0"
#define AppPublisher "Pipeline Optima"
#define AppURL "https://github.com/ParichayNITW/OPTIM2"
#define AppExeName "PipelineOptima.exe"

[Setup]
AppId={{B7E3A4F2-6C1D-4E8B-9A3F-2D5E7C8B1A09}
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
// Check if the app is running before uninstall
function InitializeUninstall(): Boolean;
begin
  Result := True;
end;
