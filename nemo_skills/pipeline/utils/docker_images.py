import hashlib
import logging
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

_DOCKERFILE_PREFIX = "dockerfile:"
_BUILT_CONTAINER_IMAGES: dict[str, str] = {}


@lru_cache(maxsize=1)
def _get_repo_root() -> Path | None:
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _sanitize_image_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return sanitized or "image"


def _docker_image_exists(image: str) -> bool:
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker is required to build images from dockerfile specifications, but it was not found in PATH."
        ) from exc


def _resolve_dockerfile_path(dockerfile_path_str: str) -> Path:
    dockerfile_path = Path(dockerfile_path_str.strip())
    if dockerfile_path.is_absolute():
        resolved = dockerfile_path
    else:
        repo_root = _get_repo_root()
        candidate_paths = []
        if repo_root is not None:
            candidate_paths.append(repo_root / dockerfile_path)
        candidate_paths.append(Path.cwd() / dockerfile_path)
        for candidate in candidate_paths:
            if candidate.exists():
                resolved = candidate
                break
        else:
            raise FileNotFoundError(
                f"Dockerfile '{dockerfile_path}' not found. Checked paths: "
                + ", ".join(str(candidate) for candidate in candidate_paths)
            )
    if not resolved.exists():
        raise FileNotFoundError(f"Dockerfile '{resolved}' not found.")
    if not resolved.is_file():
        raise ValueError(f"Dockerfile path '{resolved}' is not a file.")
    return resolved.resolve()


def _build_local_docker_image(dockerfile_spec: str) -> str:
    dockerfile_path = _resolve_dockerfile_path(dockerfile_spec)
    repo_root = _get_repo_root()
    if repo_root and _is_relative_to(dockerfile_path, repo_root):
        rel_identifier = dockerfile_path.relative_to(repo_root).as_posix()
    else:
        rel_identifier = dockerfile_path.as_posix()
    image_name = f"nemo-skills-local-{_sanitize_image_component(rel_identifier)}"
    digest = hashlib.sha256(dockerfile_path.read_bytes()).hexdigest()[:12]
    image_ref = f"{image_name}:{digest}"

    if _docker_image_exists(image_ref):
        LOG.info("Using cached Docker image %s for %s", image_ref, dockerfile_path)
        return image_ref

    if repo_root and _is_relative_to(dockerfile_path, repo_root):
        context_dir = repo_root
    else:
        context_dir = dockerfile_path.parent

    LOG.info("Building Docker image %s from %s (context: %s)", image_ref, dockerfile_path, context_dir)
    try:
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(dockerfile_path),
                "-t",
                image_ref,
                str(context_dir),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker is required to build images from dockerfile specifications, but it was not found in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to build Docker image from {dockerfile_path}") from exc

    return image_ref


def resolve_container_image(container: str | list[str], cluster_config: dict) -> str | list[str]:
    if isinstance(container, list):
        return [resolve_container_image(item, cluster_config) for item in container]

    if not isinstance(container, str):
        return container

    if not container.startswith(_DOCKERFILE_PREFIX):
        return container

    if cluster_config.get("executor") != "local":
        raise ValueError("dockerfile container specifications are only supported for the local executor.")

    if container not in _BUILT_CONTAINER_IMAGES:
        dockerfile_spec = container[len(_DOCKERFILE_PREFIX) :].strip()
        if not dockerfile_spec:
            raise ValueError("dockerfile container specification must include a path.")
        _BUILT_CONTAINER_IMAGES[container] = _build_local_docker_image(dockerfile_spec)
    return _BUILT_CONTAINER_IMAGES[container]


__all__ = ["resolve_container_image"]
