"""Package exposing search strategy implementations."""

# Expose names for convenience (optional)
from .dfs import run_dfs
from .bfs import run_bfs
from .gbfs import run_gbfs
from .astar import run_astar

__all__ = ["run_dfs", "run_bfs", "run_gbfs", "run_astar"]
