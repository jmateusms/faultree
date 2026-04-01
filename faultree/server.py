from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from faultree.builder import build, compute_event_probabilities

app = FastAPI(title="Faultree API")

class ProbabilityRequest(BaseModel):
    tree: Dict[str, Any]
    probs: Optional[Dict[str, float]] = None
    ordering: Optional[List[str]] = None
    reliability: bool = False

@app.post("/analyze")
def analyze_tree(request: ProbabilityRequest):
    try:
        # Build the BDD from the tree structure
        bdd, top, expr, symb = build(request.tree, request.ordering, success_mode=request.reliability)
        
        # Compute probabilities
        prob_map = compute_event_probabilities(
            bdd, 
            request.tree, 
            request.probs,
            success_mode=request.reliability
        )
        
        return {
            "expression": expr,
            "symbolic": symb,
            "probabilities": prob_map,
            "top_event_probability": prob_map.get(str(request.tree.get("id")))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server using uvicorn."""
    uvicorn.run(app, host=host, port=port)
