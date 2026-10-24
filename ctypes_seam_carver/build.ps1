# Windows counterpart of build.sh: builds the C API dynamic library loaded by ctypes_seam_carver.py.
# The C++ core is shared by all native modules, only the binding differs.
#
# Needs the MSVC Build Tools. Runs from any PowerShell: unless it already runs in a developer
# shell, it enters the Visual Studio developer environment for the architecture of the Python
# that will load the library. From cmd.exe: powershell -ExecutionPolicy Bypass -File build.ps1
param(
    # Python whose architecture the library must match, by default the one on PATH
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'

# Windows PowerShell does not treat a failing native command as an error
function Invoke-Native([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Exit code ${LASTEXITCODE}: $Command" }
}

# The library must match the process that loads it, not the OS: x64 Python runs emulated on ARM64.
$pythonPlatform = Invoke-Native { & $Python -c 'import sysconfig; print(sysconfig.get_platform())' }
# Architecture names: [VsDevCmd -arch, VSCMD_ARG_TGT_ARCH set by a developer shell, vswhere component]
$targets = @{
    'win-arm64' = @('arm64', 'arm64', 'Microsoft.VisualStudio.Component.VC.Tools.ARM64')
    'win-amd64' = @('amd64', 'x64', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64')
    'win32'     = @('x86', 'x86', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64')
}
if (-not $targets.ContainsKey($pythonPlatform)) { throw "Unsupported Python platform: $pythonPlatform" }
$devCmdArch, $targetArch, $vcComponent = $targets[$pythonPlatform]

if ($env:VSCMD_ARG_TGT_ARCH) {
    # Already a developer shell, which may target another architecture than Python
    if ($env:VSCMD_ARG_TGT_ARCH -ne $targetArch) {
        throw "This developer shell targets $env:VSCMD_ARG_TGT_ARCH, but $Python is $pythonPlatform. " +
              'Run build.ps1 from a plain PowerShell instead.'
    }
} else {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw 'Visual Studio Installer not found, install the MSVC Build Tools first (see init.ps1).'
    }
    # -products * includes the standalone Build Tools, which vswhere skips by default
    $vs = & $vswhere -latest -prerelease -products '*' -requires $vcComponent -property installationPath
    if (-not $vs) {
        throw "No Visual Studio installation has $vcComponent, add it in the Visual Studio Installer."
    }
    $hostArch = @{ 'ARM64' = 'arm64'; 'AMD64' = 'amd64'; 'x86' = 'x86' }[$env:PROCESSOR_ARCHITECTURE]
    Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation `
        -DevCmdArguments "-arch=$devCmdArch -host_arch=$hostArch -no_logo"
}

Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force build | Out-Null
    # build.sh flags translated: /O2 for -O3, /W4 for -Wall -Wextra, /LD for -shared, /MD links the
    # same C runtime DLL as Python. Nothing like -fPIC or -fvisibility=hidden is needed: only the
    # SC_API (__declspec(dllexport)) functions are exported. Objects and the import library go to build\.
    Invoke-Native {
        cl /nologo /O2 /std:c++17 /W4 /EHsc /MD /LD /I..\seam_carver_core `
            seam_carver_c_api.cpp ..\seam_carver_core\seam_carver_impl.cpp `
            /Fo:build\ /Fe:libseam_carver.dll /link /IMPLIB:build\libseam_carver.lib
    }
    Write-Host "Built $(Join-Path $PSScriptRoot 'libseam_carver.dll') for $pythonPlatform"
} finally {
    Pop-Location
}
