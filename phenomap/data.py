"""Canonical data-manifest interface used by the public tutorials."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


MANIFEST_ENV = "PHENOMAP_TUTORIAL_CONFIG"


def _find_repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "README.md").exists() and (candidate / "step1").is_dir():
            return candidate
    raise RuntimeError("Run from inside the PhenoMap repository or pass repository_root.")


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, str]]:
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten(value, name)
        elif value is not None:
            yield name, str(value)


@dataclass(frozen=True)
class TutorialData:
    """Resolved semantic resources from a tutorial data manifest."""

    manifest_path: Path
    repository_root: Path
    _resources: Mapping[str, Path]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._resources))

    def path(self, key: str) -> Path:
        """Return a resource path without requiring it to exist."""
        try:
            return self._resources[key]
        except KeyError as error:
            raise KeyError(
                f"Tutorial resource {key!r} is not configured. Add it under resources "
                "in your tutorial data manifest."
            ) from error

    def require(self, key: str, *, kind: str = "file") -> Path:
        """Return an existing resource or raise an actionable, path-safe error."""
        path = self.path(key)
        exists = path.is_dir() if kind == "directory" else path.exists()
        if not exists:
            raise FileNotFoundError(
                f"Required tutorial resource {key!r} is unavailable. Copy "
                "configs/tutorial_data.example.yaml to a private manifest, set its "
                "semantic resource path, and optionally select it with "
                f"{MANIFEST_ENV}."
            )
        return path


def load_tutorial_data(
    manifest_path: str | Path | None = None,
    *,
    repository_root: str | Path | None = None,
) -> TutorialData:
    """Load the selected tutorial manifest using the documented resolution order."""
    root = Path(repository_root).resolve() if repository_root else _find_repository_root()
    explicit = manifest_path or os.environ.get(MANIFEST_ENV)
    if explicit:
        selected = Path(explicit).expanduser().resolve()
    else:
        local = root / "configs" / "tutorial_data.local.yaml"
        selected = local if local.exists() else root / "configs" / "tutorial_data.example.yaml"
    if not selected.exists():
        raise FileNotFoundError(
            "No tutorial data manifest was found. Start from "
            "configs/tutorial_data.example.yaml and set PHENOMAP_TUTORIAL_CONFIG."
        )

    document = yaml.safe_load(selected.read_text(encoding="utf-8")) or {}
    resources = document.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("The tutorial manifest must contain a resources mapping.")
    resolved: dict[str, Path] = {}
    for key, value in _flatten(resources):
        path = Path(value).expanduser()
        resolved[key] = path.resolve() if path.is_absolute() else (root / path).resolve()
    return TutorialData(selected, root, resolved)
