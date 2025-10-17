import hashlib
import logging
import re
import subprocess
from pathlib import Path

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

_DOCKERFILE_PREFIX = "dockerfile:"


def _sanitize_image_component(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return sanitized


def _resolve_dockerfile_path(dockerfile_path_str: str) -> Path:
    dockerfile_path = Path(dockerfile_path_str.strip())
    if dockerfile_path.is_absolute():
        resolved = dockerfile_path
    else:
        repo_root = Path(__file__).resolve().parents[3]
        candidate_paths = []
        candidate_paths.append(Path.cwd() / dockerfile_path)
        candidate_paths.append(repo_root / dockerfile_path)
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
    rel_identifier = dockerfile_path.as_posix()
    image_name = f"locally-built-{_sanitize_image_component(rel_identifier)}"
    digest = hashlib.sha256(dockerfile_path.read_bytes()).hexdigest()[:12]
    image_ref = f"{image_name}:{digest}"
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


def resolve_container_image(container: str, cluster_config: dict) -> str:
    if not container.startswith(_DOCKERFILE_PREFIX):
        return container

    if cluster_config["executor"] != "local":
        raise ValueError("dockerfile container specifications are only supported for the local executor.")

    dockerfile_spec = container[len(_DOCKERFILE_PREFIX) :].strip()
    if not dockerfile_spec:
        raise ValueError("dockerfile container specification must include a path.")
    return _build_local_docker_image(dockerfile_spec)
