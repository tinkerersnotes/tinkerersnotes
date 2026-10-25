"""Configures this clone of the repo to commit Jupyter notebooks without their outputs.

.gitattributes applies the nbstripout filter to *.ipynb files, but git keeps filter definitions
in .git/config, which is never shared: every clone has to run this once. init.sh and init.ps1 do,
with the Python of the repo's virtual environment, where nbstripout is installed.

The filter runs that Python by its absolute path, so it works from any shell, whether the virtual
environment is activated or not. This is the manual setup from nbstripout's README, plus:
  --keep-id   keeps the cell ids Jupyter assigns, which never change, instead of renumbering
              every cell below an inserted one in each diff
  extrakeys   drops the Python version from notebook metadata, which differs between machines
"""
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def git_config(key, value):
    subprocess.run(['git', 'config', key, value], check=True)


def main():
    if shutil.which('git') is None:
        print('git not found, notebook outputs will not be stripped from commits')
        return
    if subprocess.run(['git', 'rev-parse', '--git-dir'], capture_output=True).returncode != 0:
        print('Not a git clone, notebook outputs will not be stripped from commits')
        return

    # git runs filters through sh on every platform, Git for Windows included: forward slashes
    # and shell quoting keep the interpreter path intact, with spaces or backslashes in it.
    # Not resolved: a virtual environment's python is a symlink to an interpreter without nbstripout.
    python = shlex.quote(Path(sys.executable).as_posix())
    git_config('filter.nbstripout.clean', f'{python} -m nbstripout --keep-id')
    git_config('filter.nbstripout.smudge', 'cat')
    # A failing filter makes git add fail, instead of committing the notebook with its outputs
    git_config('filter.nbstripout.required', 'true')
    git_config('filter.nbstripout.extrakeys', 'metadata.language_info.version')
    # git diff and git log -p show notebooks stripped too, including old commits made with outputs
    git_config('diff.ipynb.textconv', f'{python} -m nbstripout --keep-id -t')
    print(f'Notebook outputs are stripped from commits by: {python} -m nbstripout')


if __name__ == '__main__':
    main()
