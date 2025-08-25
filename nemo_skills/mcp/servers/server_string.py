# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="String MCP Server")

TOOLS = {
    "concat": {
        "description": "Concatenate strings",
        "input_schema": {
            "type": "object",
            "title": "ConcatInput",
            "properties": {"x": {"type": "string"}, "y": {"type": "string"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    },
}


class ToolCallRequest(BaseModel):
    tool: str
    args: Dict[str, Any]


@app.get("/list_tools")
async def list_tools():
    return [{"server": "string", "name": name, **meta} for name, meta in TOOLS.items()]


@app.post("/call_tool")
async def call_tool(request: ToolCallRequest):
    tool_name = request.tool
    args = request.args
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    # Simulate execution
    if tool_name == "concat":
        result = args["x"] + args["y"]
    else:
        result = None
    return result
