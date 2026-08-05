import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
from faultree.builder import build, compute_event_probabilities, normalize_tree

logger = logging.getLogger("faultree.server")

app = FastAPI(title="Faultree API")

class ProbabilityRequest(BaseModel):
    tree: Dict[str, Any]
    probs: Optional[Dict[str, Union[float, List[float]]]] = None
    ordering: Optional[List[str]] = None
    reliability: bool = False
    shuffle: bool = False
    seed: Optional[int] = 0
    allow_missing: bool = False


def _to_native(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-serializable native types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_tree(request: ProbabilityRequest):
    try:
        tree = normalize_tree(request.tree)

        bdd, top, expr, symb = build(tree, request.ordering, success_mode=request.reliability)

        prob_map = compute_event_probabilities(
            bdd,
            tree,
            request.probs,
            success_mode=request.reliability,
            shuffle=request.shuffle,
            seed=request.seed,
            allow_missing=request.allow_missing,
        )
        prob_map = {k: _to_native(v) for k, v in prob_map.items()}

        return {
            "expression": expr,
            "symbolic": symb,
            "probabilities": prob_map,
            "top_event_probability": prob_map.get(str(tree.get("id")))
        }
    except (ValueError, KeyError, TypeError) as e:
        # The engine raises ValueError for input problems; KeyError/TypeError
        # are kept in the client-error class as a safety net for malformed
        # payload shapes the validators miss.
        raise HTTPException(status_code=400, detail=str(e) or repr(e))
    except Exception:
        logger.exception("Unexpected error while analyzing tree")
        raise HTTPException(status_code=500, detail="Internal server error")

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the FastAPI server using uvicorn."""
    uvicorn.run(app, host=host, port=port)
