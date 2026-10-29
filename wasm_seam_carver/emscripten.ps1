# Windows counterpart of emscripten.sh: puts the Emscripten pinned there (EMSCRIPTEN_VERSION) on
# PATH: the emcc already there if it is that version, otherwise this repo's own emsdk in .emsdk\
# (git-ignored), installed on first use: about 1 GB and a few minutes, once. Dot-source it:
#
#   . wasm_seam_carver\emscripten.ps1
#
# $env:EMSDK_DIR installs emsdk elsewhere, e.g. to share one between clones.
# Needs git and Python on PATH (see init.ps1).

# The version lives in emscripten.sh only, which CI uses too
$EmscriptenVersion = (Select-String -Path (Join-Path $PSScriptRoot 'emscripten.sh') `
                                    -Pattern '^EMSCRIPTEN_VERSION=(\S+)').Matches[0].Groups[1].Value

# emcc --version: "emcc (...) 6.0.3 (commit)" for releases, "6.0.3-git (commit)" built from source
function Test-PinnedEmscripten {
    if (-not (Get-Command emcc -ErrorAction SilentlyContinue)) { return $false }
    $ErrorActionPreference = 'Continue'  # emcc may write to stderr
    $first = & emcc --version 2>$null | Select-Object -First 1
    return [bool]($first -match " $([regex]::Escape($EmscriptenVersion))(-git)? \(")
}

if (-not (Test-PinnedEmscripten)) {
    $emsdk = if ($env:EMSDK_DIR) { $env:EMSDK_DIR } else { Join-Path (Split-Path $PSScriptRoot) '.emsdk' }
    if (-not (Test-Path (Join-Path $emsdk 'emsdk.ps1'))) {
        Write-Host "Installing Emscripten $EmscriptenVersion into $emsdk (once)"
        & git clone --depth 1 https://github.com/emscripten-core/emsdk.git $emsdk
        if ($LASTEXITCODE -ne 0) { throw 'git clone of emsdk failed' }
    }
    # emsdk knows the versions released before its checkout: pull once if this one is newer
    & (Join-Path $emsdk 'emsdk.ps1') install $EmscriptenVersion
    if ($LASTEXITCODE -ne 0) {
        & git -C $emsdk pull --ff-only
        if ($LASTEXITCODE -eq 0) { & (Join-Path $emsdk 'emsdk.ps1') install $EmscriptenVersion }
        if ($LASTEXITCODE -ne 0) { throw "emsdk could not install Emscripten $EmscriptenVersion" }
    }
    & (Join-Path $emsdk 'emsdk.ps1') activate $EmscriptenVersion | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "emsdk could not activate Emscripten $EmscriptenVersion" }
    $env:EMSDK_QUIET = '1'
    try {
        . (Join-Path $emsdk 'emsdk_env.ps1') | Out-Null
    } finally {
        Remove-Item Env:EMSDK_QUIET -ErrorAction SilentlyContinue
    }
    if (-not (Test-PinnedEmscripten)) {
        throw "emcc is not Emscripten $EmscriptenVersion after setup"
    }
}
