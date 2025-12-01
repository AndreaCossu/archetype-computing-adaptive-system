from pathlib import Path
import pickle
from typing import Dict
import numpy as np

def save_results(results: Dict, output_path: str):
    """Save results to pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_path}")


def load_results(input_path: str) -> Dict:
    """Load results from pickle file."""
    with open(input_path, 'rb') as f:
        results = pickle.load(f)
    return results


    