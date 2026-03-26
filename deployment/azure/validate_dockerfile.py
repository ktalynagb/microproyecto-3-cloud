"""Validador de sincronización entre imports Python y Dockerfile.

Compara los módulos importados en los scripts Python del pipeline con los
paquetes instalados en el Dockerfile. Ejecutar antes de cada deploy para
detectar dependencias faltantes.

Uso:
    python deployment/azure/validate_dockerfile.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Mapeo de nombre de import → nombre de paquete pip
IMPORT_TO_PACKAGE: dict[str, str] = {
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "roboflow": "roboflow",
    "ultralytics": "ultralytics",
    "mlflow": "mlflow",
    "azureml": "azureml-core",
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pandas": "pandas",
    "reportlab": "reportlab",
    "requests": "requests",
    "datasets": "datasets",
    "huggingface_hub": "huggingface-hub",
    "azure": "azure-ai-ml",
    "dotenv": "python-dotenv",
}

# Módulos de la biblioteca estándar que no requieren instalación pip
STDLIB_MODULES = {
    "argparse", "ast", "collections", "contextlib", "dataclasses", "datetime",
    "enum", "functools", "hashlib", "io", "itertools", "json", "logging",
    "math", "os", "pathlib", "pickle", "random", "re", "shutil", "signal",
    "socket", "subprocess", "sys", "tempfile", "threading", "time", "typing",
    "unittest", "urllib", "uuid", "warnings", "zipfile", "__future__",
    "abc", "base64", "copy", "csv", "gc", "gzip", "hashlib", "hmac",
    "html", "http", "inspect", "operator", "platform", "pprint",
    "queue", "struct", "textwrap", "traceback", "types", "weakref",
}

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DOCKERFILE = Path(__file__).resolve().parent / "Dockerfile"
_COMPONENTS_DIR = Path(__file__).resolve().parent / "components"
_BATCH_DIR = Path(__file__).resolve().parent / "batch_inference"


def _extract_imports_from_file(path: Path) -> set[str]:
    """Extrae los módulos de primer nivel importados en un archivo Python."""
    imports: set[str] = set()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, OSError):
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    return imports - STDLIB_MODULES


def _extract_packages_from_dockerfile(dockerfile: Path) -> set[str]:
    """Extrae los paquetes pip instalados en el Dockerfile."""
    packages: set[str] = set()
    if not dockerfile.exists():
        return packages

    content = dockerfile.read_text(encoding="utf-8")
    # Normalizar continuaciones de línea
    content_joined = re.sub(r"\\\n", " ", content)

    # Buscar todas las líneas que sigan a 'pip install'
    in_pip = False
    for line in content_joined.splitlines():
        stripped = line.strip()
        if re.search(r"pip install", stripped, re.IGNORECASE):
            in_pip = True
        if in_pip:
            # Extraer tokens que sean nombres de paquetes
            tokens = stripped.split()
            for token in tokens:
                if token.startswith("-") or token in (
                    "pip", "install", "--no-cache-dir", "--upgrade",
                    "RUN", "&&", "||"
                ):
                    continue
                # Extraer nombre base (antes de >=, ==, <=, etc.)
                pkg = re.split(r"[><=!@;]", token)[0].strip()
                if pkg and not pkg.startswith("#"):
                    packages.add(pkg.lower())
            # Terminar el bloque pip si la línea no tiene continuación
            if not stripped.endswith("\\") and in_pip:
                in_pip = False

    return packages


def _get_local_module_names(directories: list[Path]) -> set[str]:
    """Retorna los nombres de módulos locales (archivos .py en los directorios)."""
    local_modules: set[str] = set()
    for d in directories:
        for py_file in d.glob("*.py"):
            local_modules.add(py_file.stem)
    return local_modules


def validate(
    components_dirs: list[Path],
    dockerfile: Path,
    verbose: bool = True,
) -> bool:
    """Ejecuta la validación de sincronización.

    Retorna True si no hay dependencias faltantes, False si hay problemas.
    """
    # Recopilar todos los imports de los scripts Python
    all_imports: dict[str, set[str]] = {}  # import_name → {archivos que lo usan}
    local_modules = _get_local_module_names(components_dirs)

    for comp_dir in components_dirs:
        for py_file in comp_dir.glob("*.py"):
            file_imports = _extract_imports_from_file(py_file)
            for imp in file_imports:
                # Excluir módulos locales del mismo proyecto
                if imp in local_modules:
                    continue
                all_imports.setdefault(imp, set()).add(py_file.name)

    # Extraer paquetes del Dockerfile
    dockerfile_packages = _extract_packages_from_dockerfile(dockerfile)

    if verbose:
        print(f"\n{'='*60}")
        print("VALIDACIÓN: Imports Python ↔ Dockerfile")
        print(f"{'='*60}")
        print(f"Dockerfile: {dockerfile}")
        print(f"Paquetes en Dockerfile: {len(dockerfile_packages)}")
        print(f"Módulos Python importados: {len(all_imports)}")
        print()

    # Verificar que cada import externo esté cubierto por el Dockerfile
    missing: list[tuple[str, str, set[str]]] = []
    for import_name, used_by in sorted(all_imports.items()):
        # Mapear import → paquete pip
        pip_name = IMPORT_TO_PACKAGE.get(import_name, import_name).lower()
        # Verificar si está en el Dockerfile (nombre del paquete)
        if pip_name not in dockerfile_packages and import_name.lower() not in dockerfile_packages:
            missing.append((import_name, pip_name, used_by))

    if missing:
        print("❌ DEPENDENCIAS FALTANTES EN DOCKERFILE:")
        for import_name, pip_name, used_by in missing:
            files = ", ".join(sorted(used_by))
            print(f"  - import '{import_name}' (pip: {pip_name}) → usado en: {files}")
        print(
            "\nSolución: Agrega los paquetes faltantes al Dockerfile con:\n"
            "  pip install --no-cache-dir <paquete>>=<version>"
        )
        return False

    if verbose:
        print("✅ Todos los imports Python están cubiertos en el Dockerfile.")
    return True


def main() -> None:
    components_dirs = [d for d in [_COMPONENTS_DIR, _BATCH_DIR] if d.exists()]
    success = validate(components_dirs, _DOCKERFILE, verbose=True)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
