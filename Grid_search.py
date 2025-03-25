import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class PINNGridSearch:
    def __init__(self):
        self.results = []
        self.best = {'params': None, 'loss': float('inf')}
    
    def train_model(self, params, X, y):
        model = PINN(X.shape[1], params['hidden_size'], params['num_layers'])
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=params['decay_step'], 
                                            gamma=params['decay_factor'])
        criterion = nn.MSELoss()
        
        losses = []
        for epoch in range(params['epochs']):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        
        return losses[-1], losses

    def search(self, X, y, param_grid):
        param_names = param_grid.keys()
        param_values = param_grid.values()
        total_combinations = len(list(product(*param_values)))
        
        with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
            for params in product(*param_values):
                param_dict = dict(zip(param_names, params))
                final_loss, epoch_losses = self.train_model(param_dict, X, y)
                
                self.results.append({
                    **param_dict,
                    'final_loss': final_loss,
                    'epoch_losses': epoch_losses
                })
                
                if final_loss < self.best['loss']:
                    self.best['loss'] = final_loss
                    self.best['params'] = param_dict
                
                pbar.update(1)
                pbar.set_postfix({'best_loss': self.best['loss']})
        
        return pd.DataFrame(self.results), self.best

# Your specified hyperparameter space
param_grid = {
    'num_layers': [2, 3, 4],            # Number of Hidden Layers
    'hidden_size': [16, 32, 63],        # Hidden Layer Size (using 63 instead of 63)
    'epochs': [100, 1000, 2000],        # Number of Epochs
    'lr': [0.001, 0.01, 0.1],          # Learning Rate
    'decay_step': [10, 20, 100],        # Learning Rate Decay Step
    'decay_factor': [0.1, 0.5, 0.9]     # Learning Rate Decay Factor
}

if __name__ == "__main__":
    # Generate dummy data (replace with your actual data)
    X = torch.randn(1000, 2)  # 1000 samples, 2 features
    y = torch.randn(1000, 1)  # 1000 targets
    
    # Run exhaustive grid search
    print("Starting grid search with all possible combinations...")
    searcher = PINNGridSearch()
    results_df, best_result = searcher.search(X, y, param_grid)
    
    # Save results
    results_df.to_csv('pinn_grid_search_results.csv', index=False)
    
    # Print best results
    print("\nBest parameters found:")
    for param, value in best_result['params'].items():
        print(f"{param:>20}: {value}")
    print(f"{'Final validation loss':>20}: {best_result['loss']:.6f}")
    
    # Plot top 5 configurations
    top_5 = results_df.sort_values('final_loss').head(5)
    plt.figure(figsize=(12, 7))
    for _, row in top_5.iterrows():
        label = (f"Layers: {row['num_layers']}, Size: {row['hidden_size']}\n"
                f"LR: {row['lr']}, Epochs: {row['epochs']}\n"
                f"Step: {row['decay_step']}, Factor: {row['decay_factor']}")
        plt.plot(row['epoch_losses'], label=label)
    
    plt.yscale('log')
    plt.title('Training Loss Curves (Top 5 Configurations)', pad=20)
    plt.xlabel('Epoch', labelpad=10)
    plt.ylabel('Loss (log scale)', labelpad=10)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('top_5_configurations.png', dpi=300, bbox_inches='tight')
    plt.show()