from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from research.pipeline.control.residual.models.models import NeuralFlyModel

def create_dataset(states, thrusts, forces, torques, wind_vector):
    # For neural fly, we just need the current state, current action, target force/torque, and wind vector.
    X_state = states[:-1] # Because states is length N+1, thrusts is N
    X_action = thrusts
    Y = np.concatenate([forces, torques], axis=-1)
    # Wind vector is constant for this trajectory, repeat it for all steps
    W = np.tile(wind_vector, (len(X_state), 1))
    
    return X_state, X_action, Y, W

def main():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
        
    config_path = Path("configs/research/modules/px4_config/px4_hover.toml")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        
    res_cfg = config.get("residual_model", {})
        
    dr_data_dir = Path("research/modules/data/domain_randomization_400")
    all_states, all_actions, all_targets, all_winds = [], [], [], []
    
    if dr_data_dir.exists():
        print(f"Loading Domain Randomization datasets from {dr_data_dir}")
        for npz_file in dr_data_dir.glob("*.npz"):
            data = np.load(npz_file)
            
            wind_vector = data.get("wind_vector")
            if wind_vector is None:
                wind_vector = np.array([0.0, 0.0, 0.0])
                
            s, a, y, w = create_dataset(
                data["states"], data["thrusts"], 
                data["disturbance_forces"], data["disturbance_torques"], 
                wind_vector
            )
            if len(s) > 0:
                all_states.append(s)
                all_actions.append(a)
                all_targets.append(y)
                all_winds.append(w)
            
    if not all_states:
        print("No DR data found.")
        return
        
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    winds = np.concatenate(all_winds, axis=0)
    
    # Save parameters for TSNE later
    tsne_data_path = Path("research/modules/neural_fly_adaptive/runs/tsne_data.npz")
    tsne_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tsne_data_path, states=states, actions=actions, targets=targets, winds=winds)
    
    state_noise_std = res_cfg.get("train_state_noise_std", 0.05)
    noisy_states = states + np.random.normal(0, state_noise_std, states.shape)
    
    target_noise_std = res_cfg.get("train_target_noise_std", 0.05)
    noisy_targets = targets + np.random.normal(0, target_noise_std, targets.shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    X_s = torch.tensor(noisy_states, dtype=torch.float32).to(device)
    X_a = torch.tensor(actions, dtype=torch.float32).to(device)
    Y = torch.tensor(noisy_targets, dtype=torch.float32).to(device)
    W = torch.tensor(winds, dtype=torch.float32).to(device)
    
    # Shuffle data
    indices = torch.randperm(X_s.shape[0], device=device)
    X_s = X_s[indices]
    X_a = X_a[indices]
    Y = Y[indices]
    W = W[indices]
    
    # Train test split
    split = int(0.8 * X_s.shape[0])
    Xs_train, Xs_test = X_s[:split], X_s[split:]
    Xa_train, Xa_test = X_a[:split], X_a[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    W_train, W_test = W[:split], W[split:]
    
    model = NeuralFlyModel(state_dim=13, action_dim=4, wind_dim=3, num_basis=res_cfg.get("num_basis", 16), output_dim=6).to(device)
    learning_rate = res_cfg.get("train_learning_rate", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    log_dir = "research/modules/neural_fly_adaptive/tensorboard_logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    epochs = 500  # Faster training
    batch_size = 512  # Faster batches
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(Xs_train.shape[0], device=device)
        
        epoch_loss = 0
        for i in range(0, Xs_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_xs = Xs_train[indices]
            batch_xa = Xa_train[indices]
            batch_w = W_train[indices]
            batch_y = Y_train[indices]
            
            optimizer.zero_grad()
            pred = model(batch_xs, batch_xa, batch_w)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_xs.shape[0]
            
        epoch_loss /= Xs_train.shape[0]
        
        if epoch % max(1, epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(Xs_test, Xa_test, W_test)
                test_loss = criterion(test_pred, Y_test)
            print(f"Neural-Fly Epoch {epoch:4d} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss.item():.6f}")
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Loss/test", test_loss.item(), epoch)
            
    writer.close()
    print(f"Final Neural-Fly Train Loss: {epoch_loss:.6f}")
    
    model_save_path = Path("research/modules/neural_fly_adaptive/models/residual_model.pt")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the CPU state dict
    torch.save(model.cpu().state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
