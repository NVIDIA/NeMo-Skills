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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Math MCP Server")

TOOLS = {
    "add": {
        "description": "Add two numbers",
        "input_schema": {
            "type": "object",
            "title": "AddInput",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"],
            "additionalProperties": False
        },
    },
    "mul": {
        "description": "Multiply two numbers",
        "input_schema": {
            "type": "object",
            "title": "MultiplyInput",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"],
            "additionalProperties": False
        },
    },
}

class ToolCallRequest(BaseModel):
    tool: str
    args: Dict[str, Any]

@app.get("/tools")
async def list_tools():
    return [{"server": "math", "name": name, **meta} for name, meta in TOOLS.items()]

@app.post("/call")
async def call_tool(request: ToolCallRequest):
    tool_name = request.tool
    args = request.args
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    # Simulate execution
    if tool_name == "add":
        result = args["a"] + args["b"] + 2
    elif tool_name == "mul":
        result = args["a"] * args["b"] * 2
    else:
        result = None
    return result
