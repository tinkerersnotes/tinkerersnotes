# Puts Emscripten EMSCRIPTEN_VERSION on PATH: the emcc already there if it is that version,
# otherwise this repo's own emsdk in .emsdk/ (git-ignored), installed on first use: about 1 GB and
# a few minutes, once. build.sh and ../wasmtime_seam_carver/build.sh source this; to use emcc in a
# shell, source it from the repo's root:
#
#   . wasm_seam_carver/emscripten.sh
#
# EMSDK_DIR=/some/path installs emsdk there instead, e.g. to share one between clones.
# Needs git and python3 (macOS: the Xcode command line tools; Ubuntu: git python3).
#
# POSIX sh, and sourced: it returns instead of exiting, so it can't end the caller's shell.

EMSCRIPTEN_VERSION=6.0.3  # the version the page is developed and tested with; CI uses this too

# emcc --version: "emcc (...) 6.0.3 (commit)" for releases, "6.0.3-git (commit)" built from source
emscripten_is_pinned() {
    emcc --version 2>/dev/null | head -n 1 | grep -Eq " $EMSCRIPTEN_VERSION(-git)? \("
}

emscripten_setup() {
    if emscripten_is_pinned; then
        return 0  # already on PATH
    fi
    # The repo's root: sourced from it or from one of its folders, like the build scripts do
    if [ -f wasm_seam_carver/emscripten.sh ]; then
        _root=$(pwd)
    elif [ -f ../wasm_seam_carver/emscripten.sh ]; then
        _root=$(cd .. && pwd)
    else
        _root=$(git rev-parse --show-toplevel 2>/dev/null) || _root=$(pwd)
    fi
    _emsdk=${EMSDK_DIR:-$_root/.emsdk}
    if [ ! -x "$_emsdk/emsdk" ]; then
        echo "Installing Emscripten $EMSCRIPTEN_VERSION into $_emsdk (once)" >&2
        git clone --depth 1 https://github.com/emscripten-core/emsdk.git "$_emsdk" >&2 || return 1
    fi
    # emsdk knows the versions released before its checkout: pull once if this one is newer
    if ! "$_emsdk/emsdk" install "$EMSCRIPTEN_VERSION" >&2; then
        git -C "$_emsdk" pull --ff-only >&2 && "$_emsdk/emsdk" install "$EMSCRIPTEN_VERSION" >&2 || return 1
    fi
    "$_emsdk/emsdk" activate "$EMSCRIPTEN_VERSION" >/dev/null || return 1
    # emsdk_env.sh finds itself only in bash, zsh and ksh; from its own folder in any shell (dash)
    _here=$(pwd)
    export EMSDK_QUIET=1  # read by emsdk.py, so exported
    cd "$_emsdk" && . ./emsdk_env.sh >/dev/null
    _sourced=$?
    unset EMSDK_QUIET
    cd "$_here" || return 1
    [ "$_sourced" -eq 0 ] || return 1
    if ! emscripten_is_pinned; then
        echo "emcc is not Emscripten $EMSCRIPTEN_VERSION after setup: $(emcc --version 2>&1 | head -n 1)" >&2
        return 1
    fi
}

emscripten_setup
_status=$?
unset -f emscripten_setup emscripten_is_pinned
unset _root _emsdk _here _sourced
return $_status
