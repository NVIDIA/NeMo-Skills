#!/usr/bin/env bash

##
# Build and Push images to Gitlab container registry.
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

__src_dir="$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"

__dockerfile="${1}"

__context_dir=$(git rev-parse --show-toplevel 2>/dev/null)

__repo_user=$(basename $(dirname $(git remote get-url origin | sed -E -e "s|[@:]|/|g")))
__repo_name=$(basename -s .git $(git remote get-url origin))
__project_name=$(basename "${__dockerfile}")
if [[ "${__project_name}" == *.* ]]; then
    __project_name=/$(echo "${__project_name}" | cut -d. -f2)
else
    unset __project_name
fi

__docker_name=$(echo ${DOCKER_NAME:-"gitlab-master.nvidia.com/${__repo_user}/${__repo_name}${__project_name}"} | tr "[:upper:]" "[:lower:]")
__docker_tag=${DOCKER_TAG:-"$(date +"%Y.%m.%d")-$(git rev-parse --short HEAD)"}

echo "Building ${__docker_name}:${__docker_tag} from context ${__context_dir}"

if [[ ! -z ${DOCKER_PUSH} ]]; then
    __docker_build_args="${__docker_build_args} --push"
fi
if [[ ! -z ${DOCKER_CACHE} ]]; then
    __docker_build_args="${__docker_build_args} --cache-to type=registry,ref=${__docker_name}/cache,mode=max --cache-from type=registry,ref=${__docker_name}/cache"
fi

docker build ${__docker_build_args} \
    -f "${__dockerfile}" \
    -t "${__docker_name}:${__docker_tag}" \
    "${__context_dir}"
