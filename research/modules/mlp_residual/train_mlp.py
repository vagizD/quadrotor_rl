from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from research.pipeline.control.residual.models.models import MLPResidualModel

def create_history_dataset(states, thrusts, targets, forces, torques, history_len=10):
    N = len(thrusts)
    X = []
    Y = []
    
    for t in range(history_len, N):
        hist_s = states[t-history_len:t]
        hist_a = thrusts[t-history_len:t]
        hist_e = targets[t-history_len:t] - states[t-history_len:t, :3]
        
        hist_vec = np.concatenate([hist_s, hist_a, hist_e], axis=-1).flatten()
        y_vec = np.concatenate([forces[t], torques[t]], axis=-1)
        
        X.append(hist_vec)
        Y.append(y_vec)
        
    if len(X) == 0:
        return np.array([]), np.array([])
        
    return np.array(X), np.array(Y)

def main():
    dr_data_dir = Path("research/modules/data/domain_randomization_400")
    all_X, all_Y = [], []
    
    if dr_data_dir.exists():
        print(f"Loading Domain Randomization datasets from {dr_data_dir}")
        for npz_file in dr_data_dir.glob("*.npz"):
            data = np.load(npz_file)
            x, y = create_history_dataset(
                data["states"], data["thrusts"], data["target_positions"],
                data["disturbance_forces"], data["disturbance_torques"], 
                history_len=10
            )
            if len(x) > 0:
                all_X.append(x)
                all_Y.append(y)
            
    if not all_X:
        print("No DR data found.")
        return
        
    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    
    print(f"Dataset shape: X={X.shape}, Y={Y.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    
    split = int(0.8 * X.shape[0])
    X_train, X_test = X_tensor[:split], X_tensor[split:]
    Y_train, Y_test = Y_tensor[:split], Y_tensor[split:]
    
    model = MLPResidualModel(state_dim=X.shape[1], action_dim=0, wind_dim=0, output_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    log_dir = "research/modules/mlp_residual/tensorboard_logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    epochs = 500
    batch_size = 512
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0], device=device)
        
        epoch_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train[indices]
            batch_y = Y_train[indices]
            
            optimizer.zero_grad()
            pred = model(batch_x, torch.empty(batch_x.shape[0], 0, device=device), torch.empty(batch_x.shape[0], 0, device=device))
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.shape[0]
            
        epoch_loss /= X_train.shape[0]
        
        if epoch % max(1, epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test, torch.empty(X_test.shape[0], 0, device=device), torch.empty(X_test.shape[0], 0, device=device))
                test_loss = criterion(test_pred, Y_test)
            print(f"MLP Epoch {epoch:4d} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss.item():.6f}")
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Loss/test", test_loss.item(), epoch)
            
    writer.close()
    print(f"Final MLP Train Loss: {epoch_loss:.6f}")
    
    model_save_path = Path("research/modules/mlp_residual/models/mlp_residual_model.pt")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
