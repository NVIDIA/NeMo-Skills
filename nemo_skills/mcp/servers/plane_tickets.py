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

# Example motivated by https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/snippets/servers/streamable_starlette_mount.py
from __future__ import annotations

import uuid
from typing import Any
import contextlib

from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.fastmcp import FastMCP


# Initialize FastMCP server
mcp = FastMCP(name="plane_tickets", stateless_http=True)


# Simple city graph for demo purposes
CITIES: dict[str, list[str]] = {
    "SFO": ["JFK", "SEA", "LAX", "DFW"],
    "JFK": ["SFO", "LAX", "ORD", "MIA"],
    "LAX": ["SFO", "JFK", "SEA"],
    "SEA": ["SFO", "LAX"],
    "ORD": ["JFK", "DFW"],
    "DFW": ["SFO", "ORD"],
    "MIA": ["JFK"],
}


@mcp.tool()
def list_available_cities() -> list[str]:
    """List available origin cities (IATA codes)."""
    return sorted(CITIES.keys())


@mcp.tool()
def list_destination_cities(source: str) -> list[str]:
    """List valid destination cities from a given source city.

    Args:
        source: IATA code of the starting city (e.g., SFO)
    """
    src = source.upper()
    if src not in CITIES:
        raise ValueError("Unknown source city")
    return sorted(CITIES[src])


@mcp.tool()
def book_trip(starting_city: str, destination_city: str, round_trip: bool = False) -> dict[str, Any]:
    """Book a simple trip between two cities.

    Args:
        starting_city: IATA code for origin (e.g., SFO)
        destination_city: IATA code for destination (e.g., JFK)
        round_trip: Whether to include a return segment

    Returns:
        Booking confirmation with an id and itinerary segments.
    """
    src = starting_city.upper()
    dst = destination_city.upper()
    if src not in CITIES:
        raise ValueError("Unknown starting city")
    if dst not in CITIES[src]:
        raise ValueError("Destination not reachable from starting city")

    booking_id = uuid.uuid4().hex[:8].upper()
    segments: list[dict[str, str]] = [{"from": src, "to": dst}]
    if round_trip:
        segments.append({"from": dst, "to": src})

    return {
        "booking_id": booking_id,
        "round_trip": round_trip,
        "itinerary": segments,
    }


# (No cancellation or hold flows in this simplified demo)


# Expose both the raw MCP ASGI app and a Starlette-mounted app
mcp_app = mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(mcp.session_manager.run())
        yield


# Starlette application mounting the MCP server at /plane
app = Starlette(
    routes=[
        Mount("/plane", mcp_app),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    # Run as stdio MCP server
    mcp.run(transport="stdio")


