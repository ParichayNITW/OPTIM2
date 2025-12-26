[Setup]
AppName=Pipeline Optimizer
AppVersion=1.0
DefaultDirName={pf}\PipelineOptimizer
DefaultGroupName=Pipeline Optimizer
OutputDir=dist
OutputBaseFilename=PipelineOptimizerInstaller
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "dist\pipeline_optimizer.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Pipeline Optimizer"; Filename: "{app}\pipeline_optimizer.exe"
Name: "{commondesktop}\Pipeline Optimizer"; Filename: "{app}\pipeline_optimizer.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked
