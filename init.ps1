# Windows counterpart of init.sh. Sets up this repo, safe to rerun: a Python virtual environment
# with the pinned dependencies of pyproject.toml (constraints.txt), then builds the native seam
# carver modules. Run init.cmd from cmd.exe, PowerShell or Explorer: it runs this script without
# changing the PowerShell execution policy.
#
# System requirements:
#   Python >= 3.12 with its headers and import library, both included by the python.org installer:
#     winget install -e --id Python.Python.3.14
#   MSVC Build Tools with the C++ workload, which setuptools and Cython find on their own:
#     winget install -e --id Microsoft.VisualStudio.BuildTools --override "--passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$tnvenv = Join-Path $PSScriptRoot '.venv-tn'
$venvPython = Join-Path $tnvenv 'Scripts\python.exe'

# Windows PowerShell does not treat a failing native command as an error
function Invoke-Native([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Exit code ${LASTEXITCODE}: $Command" }
}

# True if Python runs Code successfully. Code must not contain double quotes, which Windows
# PowerShell does not pass to native commands intact.
function Test-Python([string]$Python, [string]$Code) {
    $ErrorActionPreference = 'Continue'  # redirected stderr would otherwise stop the script
    & $Python -c $Code 2>$null
    return $LASTEXITCODE -eq 0
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python not found, install it with: winget install -e --id Python.Python.3.14'
}
if (-not (Test-Python python 'import sys; raise SystemExit(sys.version_info < (3, 12))')) {
    throw "Python >= 3.12 is required, found $(python --version)"
}
if (-not (Test-Python python 'import os, sysconfig; raise SystemExit(not os.path.exists(os.path.join(sysconfig.get_path(''include''), ''Python.h'')))')) {
    throw 'Python headers are missing, reinstall Python from python.org or with winget'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vcTools = 'Microsoft.VisualStudio.Component.VC.Tools.ARM64', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
if (-not ((Test-Path $vswhere) -and
          (& $vswhere -latest -prerelease -products '*' -requiresAny -requires $vcTools -property installationPath))) {
    throw 'No MSVC C++ compiler found, install the Build Tools as described at the top of init.ps1'
}

# Recreate the environment if its interpreter is gone, e.g. after the base Python was upgraded or removed
if (-not ((Test-Path $venvPython) -and (Test-Python $venvPython 'pass'))) {
    Invoke-Native { python -m venv --clear $tnvenv }
}
Invoke-Native { & $venvPython -m pip install --upgrade 'pip>=25.1' }  # dependency groups need pip >= 25.1
Invoke-Native { & $venvPython -m pip install --group notebooks --group dev -c constraints.txt }

# Commit notebooks without their outputs: sets up the nbstripout git filter for this clone
Invoke-Native { & $venvPython configure_git.py }

# Native seam carver modules, imported by seam_carver.ipynb
foreach ($module in 'cy_seam_carver', 'cpp_seam_carver', 'c_seam_carver') {
    Push-Location $module
    try {
        Invoke-Native { & $venvPython setup.py build_ext --inplace }
    } finally {
        Pop-Location
    }
}
& (Join-Path $PSScriptRoot 'ctypes_seam_carver\build.ps1') -Python $venvPython

Write-Host "`nDone. Start JupyterLab with run-jupyter.cmd"
