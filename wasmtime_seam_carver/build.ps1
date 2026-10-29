# Windows counterpart of build.sh: builds seam_carver.wasm, the WebAssembly module
# wasmtime_seam_carver.py runs with Wasmtime. The flags and their reasons are in build.sh; the
# module is the same file on every platform.
#
# Needs Emscripten 6.0.3, which ..\wasm_seam_carver\emscripten.ps1 puts on PATH, installing it on
# first use (about 1 GB). Needs git and Python. From cmd.exe:
#   powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = 'Stop'

# Windows PowerShell does not treat a failing native command as an error
function Invoke-Native([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Exit code ${LASTEXITCODE}: $Command" }
}

. (Join-Path $PSScriptRoot '..\wasm_seam_carver\emscripten.ps1')

Push-Location $PSScriptRoot
try {
    Invoke-Native {
        em++ -O3 -std=c++17 -Wall -Wextra -fwasm-exceptions -sWASM_LEGACY_EXCEPTIONS=0 `
            -I../seam_carver_core -I../ctypes_seam_carver `
            ../ctypes_seam_carver/seam_carver_c_api.cpp ../seam_carver_core/seam_carver_impl.cpp `
            -sSTANDALONE_WASM --no-entry `
            '-sEXPORTED_FUNCTIONS=_sc_create,_sc_destroy,_sc_shrink_horizontally,_sc_width,_sc_height,_sc_copy_shrunk_rgb,_sc_last_error,_malloc,_free' `
            -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=2GB `
            -o seam_carver.wasm
    }
    Write-Host "Built $(Join-Path $PSScriptRoot 'seam_carver.wasm')"
} finally {
    Pop-Location
}
