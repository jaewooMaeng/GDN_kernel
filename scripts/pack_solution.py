"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Python, Triton, or CUDA) into a Solution JSON file for submission.

Special cases:
- If config language is "cuda" but the entrypoint points to a Python file
  (e.g. "binding.py::run"), pack from solution/cuda/ but emit a Python
  runnable spec so flashinfer-bench executes the Python wrapper.
- If config declares `source_files`, only those relative paths are packed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec, Solution, SourceFile

VALID_SOURCE_EXTENSIONS = {".py", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp"}


def pack_solution_tree(path: Path, spec: BuildSpec, name: str, definition: str, author: str, source_files: list[str] | None = None) -> Solution:
    sources = []
    if source_files:
        for rel_path in source_files:
            file_path = path / rel_path
            if not file_path.exists() or not file_path.is_file():
                raise FileNotFoundError(f"Configured source file not found: {file_path}")
            if file_path.suffix.lower() not in VALID_SOURCE_EXTENSIONS:
                raise ValueError(f"Unsupported source file extension: {file_path}")
            sources.append(SourceFile(path=rel_path, content=file_path.read_text(encoding='utf-8')))
    else:
        for file_path in sorted(path.rglob('*')):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in VALID_SOURCE_EXTENSIONS:
                continue
            rel_path = file_path.relative_to(path).as_posix()
            sources.append(SourceFile(path=rel_path, content=file_path.read_text(encoding='utf-8')))
    if not sources:
        raise ValueError(f"No source files found in directory: {path}")
    return Solution(name=name, definition=definition, author=author, spec=spec, sources=sources)


def load_config() -> dict:
    config_path = PROJECT_ROOT / 'config.toml'
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def pack_solution(output_path: Path = None) -> Path:
    config = load_config()
    solution_config = config['solution']
    build_config = config['build']

    language = build_config['language']
    entry_point = build_config['entry_point']
    dependencies = build_config.get('dependencies', [])
    binding = build_config.get('binding')
    source_files = build_config.get('source_files')
    entry_file = entry_point.split('::', 1)[0]
    runtime_language = 'python' if language == 'cuda' and entry_file.endswith('.py') else language

    if language == 'python':
        source_dir = PROJECT_ROOT / 'solution' / 'python'
    elif language == 'triton':
        source_dir = PROJECT_ROOT / 'solution' / 'triton'
    elif language == 'cuda':
        source_dir = PROJECT_ROOT / 'solution' / 'cuda'
    else:
        raise ValueError(f'Unsupported language: {language}')
    if not source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {source_dir}')

    dps = build_config.get('destination_passing_style', True)
    spec = BuildSpec(
        language=runtime_language,
        target_hardware=['cuda'],
        entry_point=entry_point,
        dependencies=dependencies,
        destination_passing_style=dps,
        binding=None if runtime_language == 'python' else binding,
    )

    solution = pack_solution_tree(
        path=source_dir,
        spec=spec,
        name=solution_config['name'],
        definition=solution_config['definition'],
        author=solution_config['author'],
        source_files=source_files,
    )

    if output_path is None:
        output_path = PROJECT_ROOT / 'solution.json'
    output_path.write_text(solution.model_dump_json(indent=2))
    print(f'Solution packed: {output_path}')
    print(f'  Name: {solution.name}')
    print(f'  Definition: {solution.definition}')
    print(f'  Author: {solution.author}')
    print(f'  Config language: {language}')
    print(f'  Runtime language: {runtime_language}')
    if source_files:
        print(f'  Source files: {source_files}')
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Pack solution files into solution.json')
    parser.add_argument('-o', '--output', type=Path, default=None, help='Output path for solution.json (default: ./solution.json)')
    args = parser.parse_args()
    try:
        pack_solution(args.output)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
