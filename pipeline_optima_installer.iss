; Pipeline Optima™ — Inno Setup Installer Script
;
; PREREQUISITES (do these on a Windows machine):
;   1. Install Python 3.11:        https://www.python.org/downloads/
;   2. Install PyInstaller:        pip install pyinstaller
;   3. Build the app bundle:       pyinstaller pipeline_optima.spec
;      (this creates dist\PipelineOptima\ folder)
;   4. Install Inno Setup 6:       https://jrsoftware.org/isdl.php
;   5. Open this file in Inno Setup, click Build > Compile
;   6. Find your installer at:     Output\PipelineOptima_Setup.exe
;
; The resulting Setup.exe is fully standalone — users just double-click it.
; No Python, no terminal, no technical knowledge required.

#define AppName      "Pipeline Optima"
#define AppVersion   "1.0.0"
#define AppPublisher "Your Company Name"
#define AppURL       "http://your-company-intranet/"
#define AppExeName   "PipelineOptima.exe"
#define AppPort      "8501"

[Setup]
; Unique ID for this application — regenerate at https://www.guidgenerator.com
AppId={{A3F2B1C4-9D8E-4F7A-B6C5-12345678ABCD}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
LicenseFile=
; OutputDir is where Setup.exe will be created (relative to this .iss file)
OutputDir=Output
OutputBaseFilename=PipelineOptima_Setup
; Use a compression that balances size and speed
Compression=lzma2/ultra64
SolidCompression=yes
; Require admin rights so it installs to Program Files
PrivilegesRequired=admin
; Minimum Windows version: Windows 10
MinVersion=10.0
; Show a modern wizard style
WizardStyle=modern
; Set the installer icon (must be a .ico file)
; SetupIconFile=logo.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
; Checkboxes shown during installation
Name: "desktopicon";    Description: "Create a &desktop shortcut";            GroupDescription: "Additional shortcuts:"; Flags: checked
Name: "startmenuicon";  Description: "Create a &Start Menu shortcut";         GroupDescription: "Additional shortcuts:"; Flags: checked
Name: "quicklaunch";    Description: "Add to &Quick Launch bar";               GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; Copy the entire PyInstaller output folder into the installation directory
; IMPORTANT: adjust the Source path if your dist folder is in a different location
Source: "dist\PipelineOptima\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Desktop shortcut
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
  Comment: "Launch Pipeline Optima™ optimization software"; \
  Tasks: desktopicon

; Start Menu shortcut
Name: "{group}\{#AppName}";          Filename: "{app}\{#AppExeName}"; Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}";      Tasks: startmenuicon

[Run]
; Optionally launch the app immediately after installation
Filename: "{app}\{#AppExeName}"; \
  Description: "Launch {#AppName} now"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up any files created by the app at runtime
Type: filesandordirs; Name: "{app}"

[Code]
// Optional: check that no previous instance is running before install
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

// Show a friendly completion message
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox(
      'Pipeline Optima has been installed successfully!' + #13#10 + #13#10 +
      'To use the application:' + #13#10 +
      '  1. Double-click the Pipeline Optima icon on your desktop.' + #13#10 +
      '  2. Your browser will open automatically.' + #13#10 +
      '  3. If the browser does not open, go to: http://localhost:{#AppPort}' + #13#10 + #13#10 +
      'Keep the application window open while using the software.',
      mbInformation, MB_OK
    );
  end;
end;
