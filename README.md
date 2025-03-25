# PINN Hyperparameter Grid Search

Simple grid search for ANN using PyTorch.

## How to Use

1. Install requirements for running the code

2. Run the grid search:
```bash
Grid_search.py
```

3. Check results:
- `grid_search_results.csv` - All parameter combinations and their losses
- `top_5_configurations.png` - Training curves for top 5 configurations

## Hyperparameters Tested
- Hidden layers: [2, 3, 4]
- Hidden units: [16, 32, 63]
- Learning rate: [0.001, 0.01, 0.1]
- Epochs: [100, 1000, 2000]
- LR decay step: [10, 20, 100]
- LR decay factor: [0.1, 0.5, 0.9]
