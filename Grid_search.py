# Import required libraries
import torch  # PyTorch for deep learning
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from itertools import product  # For Cartesian product of parameters
import pandas as pd  # For data handling
import matplotlib.pyplot as plt  # For visualization
from tqdm import tqdm  # For progress bars

# Define the Physics-Informed Neural Network (PINN) architecture
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PINN, self).__init__()  # Initialize parent class
        # Create network layers
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]  # Input layer
        # Add hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, 1))  # Output layer
        self.net = nn.Sequential(*layers)  # Combine all layers
    
    def forward(self, x):
        return self.net(x)  # Forward pass through the network

# Class for performing hyperparameter grid search on PINN
class PINNGridSearch:
    def __init__(self):
        self.results = []  # Store all search results
        self.best = {'params': None, 'loss': float('inf')}  # Track best configuration
    
    # Method to train a single model configuration
    def train_model(self, params, X, y):
        # Initialize model with current parameters
        model = PINN(X.shape[1], params['hidden_size'], params['num_layers'])
        # Set up optimizer and learning rate scheduler
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=params['decay_step'], 
            gamma=params['decay_factor']
        )
        criterion = nn.MSELoss()  # Loss function
        
        losses = []  # Track loss per epoch
        # Training loop
        for epoch in range(params['epochs']):
            optimizer.zero_grad()  # Clear gradients
            outputs = model(X)  # Forward pass
            loss = criterion(outputs, y)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            scheduler.step()  # Update learning rate
            losses.append(loss.item())  # Store loss
        
        return losses[-1], losses  # Return final loss and all losses

    # Method to perform grid search over parameter space
    def search(self, X, y, param_grid):
        param_names = param_grid.keys()  # Get parameter names
        param_values = param_grid.values()  # Get parameter values
        total_combinations = len(list(product(*param_values)))  # Calculate total combinations
        
        # Progress bar for tracking search progress
        with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
            # Iterate through all parameter combinations
            for params in product(*param_values):
                param_dict = dict(zip(param_names, params))  # Create parameter dictionary
                # Train model with current parameters
                final_loss, epoch_losses = self.train_model(param_dict, X, y)
                
                # Store results
                self.results.append({
                    **param_dict,
                    'final_loss': final_loss,
                    'epoch_losses': epoch_losses
                })
                
                # Update best configuration if current is better
                if final_loss < self.best['loss']:
                    self.best['loss'] = final_loss
                    self.best['params'] = param_dict
                
                pbar.update(1)  # Update progress
                pbar.set_postfix({'best_loss': self.best['loss']})  # Show best loss
        
        return pd.DataFrame(self.results), self.best  # Return results and best config

# Hyperparameter search space definition
param_grid = {
    'num_layers': [2, 3, 4],            # Number of hidden layers to try
    'hidden_size': [16, 32, 63],        # Hidden layer sizes to try
    'epochs': [100, 1000, 2000],        # Training durations to try
    'lr': [0.001, 0.01, 0.1],          # Learning rates to try
    'decay_step': [10, 20, 100],        # LR decay steps to try
    'decay_factor': [0.1, 0.5, 0.9]     # LR decay factors to try
}

# Main execution block
if __name__ == "__main__":
    # Create dummy data (replace with real data)
    X = torch.randn(1000, 2)  # 1000 samples with 2 features each
    y = torch.randn(1000, 1)  # 1000 target values
    
    # Run grid search
    print("Starting grid search with all possible combinations...")
    searcher = PINNGridSearch()
    results_df, best_result = searcher.search(X, y, param_grid)
    
    # Save results to CSV
    results_df.to_csv('pinn_grid_search_results.csv', index=False)
    
    # Print best configuration found
    print("\nBest parameters found:")
    for param, value in best_result['params'].items():
        print(f"{param:>20}: {value}")  # Right-aligned parameter names
    print(f"{'Final validation loss':>20}: {best_result['loss']:.6f}")
    
    # Visualize top 5 configurations
    top_5 = results_df.sort_values('final_loss').head(5)  # Get top 5
    plt.figure(figsize=(12, 7))  # Create figure
    # Plot loss curves for each top configuration
    for _, row in top_5.iterrows():
        label = (f"Layers: {row['num_layers']}, Size: {row['hidden_size']}\n"
                f"LR: {row['lr']}, Epochs: {row['epochs']}\n"
                f"Step: {row['decay_step']}, Factor: {row['decay_factor']}")
        plt.plot(row['epoch_losses'], label=label)
    
    plt.yscale('log')  # Log scale for better visualization
    plt.title('Training Loss Curves (Top 5 Configurations)', pad=20)
    plt.xlabel('Epoch', labelpad=10)
    plt.ylabel('Loss (log scale)', labelpad=10)
    plt.grid(True, which="both", ls="--", alpha=0.3)  # Add grid
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
    plt.tight_layout()  # Adjust layout
    plt.savefig('top_5_configurations.png', dpi=300, bbox_inches='tight')  # Save figure
    plt.show()  # Display plot
