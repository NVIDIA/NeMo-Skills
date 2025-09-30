#!/usr/bin/env bash

##
# Build and Push images.
#
# Usage:
#   ./dockerfiles/build.sh [/path/to/Dockerfile]
#
# Configuration Environment variables:
#   DOCKER_NAME: fully qualified name of the docker image (default inferred from repository)
#   DOCKER_TAG: docker tag (default set as `YY.MM.DD-git-hash`)
#   DOCKER_PUSH: pushes docker image when variable is set.
#   DOCKER_CACHE: uses registry cache when variable is set.
#

if [[ -z "${1}" ]]; then
    echo "Missing Dockerfile argument."
    echo "Usage: ./dockerfiles/build.sh [/path/to/Dockerfile]"
    exit 1
fi

__dockerfile="${1}"

##
#  Conditions to set the context for docker build.
#   1. If the Dockerfile is part of a git repo, the set to repo root.
#   2. If not part of git repo, then set to the directory of the Dockerfile.
#
__context_dir="$(dirname "$(realpath "${__dockerfile}")")"

pushd "${__context_dir}" > /dev/null

__git_repo_root=$(git rev-parse --show-toplevel 2>/dev/null)
__is_git_repo=$([[ $? -eq 0 ]] && echo 1 || echo 0)

## If not a git repo, we go back to working directory.
if [[ $__is_git_repo -eq 0 ]]; then
    popd > /dev/null
fi

if [[ -z "${DOCKER_NAME}" ]]; then
    __git_remote=$(git remote get-url origin 2>/dev/null)
    if [[ $? -ne 0 ]]; then
        echo "Dockerfile is not part of a git repo. Set DOCKER_NAME explicitly."
        exit 1
    fi

    __repo_user=$(basename "$(dirname "$(echo "${__git_remote}" | sed -E -e "s|[@:]|/|g")")")
    __repo_name=$(basename -s .git "$(echo "${__git_remote}")")
    __project_name=$(basename "${__dockerfile}")
    if [[ "${__project_name}" == *.* ]]; then
        __project_name=/$(echo "${__project_name}" | cut -d. -f2)
    else
        unset __project_name
    fi
    DOCKER_NAME="gitlab-master.nvidia.com/${__repo_user}/${__repo_name}${__project_name}"
fi
DOCKER_NAME=$(echo "${DOCKER_NAME}" | tr "[:upper:]" "[:lower:]")

if [[ -z "${DOCKER_TAG}" ]]; then
    __git_sha=$(git rev-parse --short HEAD 2>/dev/null)
    if [[ $? -ne 0 ]]; then
        echo "Dockerfile is not part of a git repo. Set DOCKER_TAG explicitly."
        exit 1
    fi
    DOCKER_TAG="$(date +"%Y.%m.%d")-${__git_sha}"
fi

if [[ ${__is_git_repo} -eq 1 ]]; then
    __context_dir="${__git_repo_root}"
    popd > /dev/null
fi

echo "Building ${DOCKER_NAME}:${DOCKER_TAG} from context ${__context_dir}"

if [[ ! -z ${DOCKER_PUSH} ]]; then
    __docker_build_args="${__docker_build_args} --push"
fi
if [[ ! -z ${DOCKER_CACHE} ]]; then
    __docker_build_args="${__docker_build_args} --cache-to type=registry,ref=${DOCKER_NAME}/cache,mode=max --cache-from type=registry,ref=${DOCKER_NAME}/cache"
fi

docker build ${__docker_build_args} \
    -f "${__dockerfile}" \
    -t "${DOCKER_NAME}:${DOCKER_TAG}" \
    "${__context_dir}"
