// Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

function registerTextarea() {
    var textareas = document.querySelectorAll("textarea");
    textareas.forEach(function(textarea) {
        function updateHeight() {
            textarea.style.height = 0 + 'px';

            var height = Math.max(textarea.scrollHeight, textarea.offsetHeight,
                            textarea.clientHeight);

            textarea.style.height = height + 'px';
        };
        textarea.onload = updateHeight;
        textarea.onresize = updateHeight;
        textarea.addEventListener('input', updateHeight);
        updateHeight()       
    });
};
