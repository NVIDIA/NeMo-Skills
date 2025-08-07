#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: needs to run from the root of the repo!

SANDBOX_NAME=${1:-'local-sandbox'}
DEPLOYMENT_MODE=${DEPLOYMENT_MODE:-'multi-worker'}

docker build --tag=${SANDBOX_NAME} --build-arg="UWSGI_PROCESSES=$((nproc --all * 10))" --build-arg="UWSGI_CHEAPER=nproc --all" -f dockerfiles/Dockerfile.sandbox .

echo "Starting sandbox in $DEPLOYMENT_MODE mode..."

if [ "$DEPLOYMENT_MODE" = "multi-worker" ]; then
    echo "Multi-worker mode: Starting $((`nproc --all` * 10)) workers with session affinity"
    docker run --network=host \
        -e DEPLOYMENT_MODE=multi-worker \
        -e NUM_WORKERS=$((`nproc --all` * 10)) \
        --rm --name=local-sandbox ${SANDBOX_NAME}
else
    echo "Single-worker mode: High-performance single container (default)"
    docker run --network=host \
        -e DEPLOYMENT_MODE=single \
        --rm --name=local-sandbox ${SANDBOX_NAME}
fi
