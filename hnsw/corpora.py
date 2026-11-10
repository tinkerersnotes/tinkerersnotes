"""Texts to search: docstrings of Python's library and packages, and this repository's notes.

Each corpus is a list of (source, text) pairs: where the text comes from, and the text to embed.
"""
import ast
import importlib.util
import json
import re
import subprocess
import sysconfig
from pathlib import Path

SKIPPED_DIRS = {'__pycache__', 'site-packages', 'test', 'tests', 'idle_test'}


def first_paragraph(docstring):
    return ' '.join(docstring.strip().split('\n\n')[0].split())


def docstrings(root, prefix='', min_length=20, seen=None):
    """Summaries of the public modules, classes and functions under a directory, from the source files.

    The files are parsed, never imported, so nothing in them runs. A text is the qualified name and the
    docstring's first paragraph, 'json.dumps: Serialize obj to a JSON formatted str.' Test directories,
    private classes and functions (a leading underscore), very short docstrings and ones already seen are
    left out. Private modules are kept: _pydatetime holds the datetime classes, for instance.
    """
    root = Path(root)
    seen = set() if seen is None else seen
    texts = []
    for path in sorted(root.rglob('*.py')):
        parts = path.relative_to(root).with_suffix('').parts
        if SKIPPED_DIRS & set(parts):
            continue
        module = '.'.join(filter(None, (prefix, *(parts[:-1] if parts[-1] == '__init__' else parts))))
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue

        def visit(node, name):
            docstring = ast.get_docstring(node)
            if docstring:
                summary = first_paragraph(docstring)
                if len(summary) >= min_length and summary not in seen:
                    seen.add(summary)
                    texts.append((name, f'{name}: {summary}'))
            for child in node.body:
                if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and not child.name.startswith('_')):
                    visit(child, f'{name}.{child.name}')

        visit(tree, module)
    return texts


def python_docstrings(packages=('numpy', 'matplotlib')):
    """Docstrings of the standard library and of installed packages, found without importing them."""
    seen = set()
    texts = docstrings(sysconfig.get_paths()['stdlib'], seen=seen)
    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec and spec.submodule_search_locations:
            texts += docstrings(spec.submodule_search_locations[0], package, seen=seen)
    return texts


def markdown_paragraphs(markdown, min_length=40):
    """Paragraphs of Markdown text without code blocks, tables and images, whitespace collapsed."""
    markdown = re.sub(r'```.*?```', '', markdown, flags=re.DOTALL)
    paragraphs = []
    for block in re.split(r'\n\s*\n', markdown):
        lines = [line for line in block.splitlines()
                 if not line.lstrip().startswith(('|', '![', '[![', '<'))]
        text = ' '.join(' '.join(lines).split())
        if len(text) >= min_length:
            paragraphs.append(text)
    return paragraphs


def repo_notes(root='..'):
    """Paragraphs of the repository's Markdown files and notebooks' Markdown cells, and its commit messages.

    Only files git tracks are read, and commits come from git log, so root must be a git checkout.
    """
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True,
                              check=True).stdout

    texts = []
    for name in git('ls-files', '*.md', '*.ipynb').split('\n'):
        if not name:
            continue
        path = Path(root, name)
        if name.endswith('.md'):
            chunks = [path.read_text(encoding='utf-8')]
        else:
            cells = json.loads(path.read_text(encoding='utf-8'))['cells']
            chunks = [''.join(c['source']) for c in cells if c['cell_type'] == 'markdown']
        texts += [(name, p) for chunk in chunks for p in markdown_paragraphs(chunk)]
    for commit in git('log', '--format=%h%x00%B%x01').split('\x01'):
        if commit.strip():
            sha, message = commit.strip().split('\x00')
            # Without the attribution trailer, which every recent commit shares
            message = '\n'.join(line for line in message.splitlines() if not line.startswith('Co-Authored-By'))
            texts.append((f'commit {sha}', ' '.join(message.split())))
    return texts
