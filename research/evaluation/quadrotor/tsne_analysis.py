import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from sklearn.manifold import TSNE
import seaborn as sns

from research.pipeline.control.residual.models.models import NeuralFlyModel

def main():
    dr_data_dir = Path("research/modules/data/domain_randomization_400")
    if not dr_data_dir.exists():
        print("Data directory not found.")
        return
        
    model_path = Path("research/modules/neural_fly_adaptive/models/residual_model.pt")
    if not model_path.exists():
        print("Neural-Fly model not found.")
        return
        
    model = NeuralFlyModel(state_dim=13, action_dim=4, wind_dim=3, num_basis=16)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    a_hats = []
    labels = []
    
    # Process trajectories to emulate online adaptation and get final a_hat
    for npz_file in dr_data_dir.glob("*.npz"):
        data = np.load(npz_file)
        
        # We can extract the "true" wind label
        wind_label = str(data["wind_label"][0]) if "wind_label" in data else "unknown"
        labels.append(wind_label)
        
        # Emulate Neural-Fly adaptation on this trajectory
        states = data["states"][:-1]
        actions = data["thrusts"]
        target_positions = data["target_positions"]
        
        a_hat = np.zeros(16)
        dt = 0.01
        lr = 0.01
        a_max = 5.0
        
        for i in range(len(states)):
            s_vec = states[i]
            target_pos = target_positions[i]
            a_vec = actions[i]
            
            state_tensor = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(a_vec, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                phi = model.basis_net(state_tensor, action_tensor).squeeze(0).numpy()
                
            phi_f = phi[:3, :]
            
            e_p = target_pos - s_vec[:3]
            e_v = -s_vec[3:6]
            s = e_v + 2.0 * e_p
            
            sigma = 0.1
            update = -lr * (phi_f.T @ s) - sigma * a_hat
            
            a_hat += update * dt
            a_hat = np.clip(a_hat, -a_max, a_max)
            
        a_hats.append(a_hat)
        
    if not a_hats:
        print("No data processed.")
        return
        
    X = np.array(a_hats)
    
    # Run t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette="viridis", s=100)
    plt.title("t-SNE of Neural-Fly Adaptive Parameters (a_hat)")
    
    # Save to artifacts dir
    out_path = Path("/home/vagiz/.gemini/antigravity-ide/brain/f1944d16-2d4c-41fc-9d63-01a9781df2f7/tsne_clusters.png")
    plt.savefig(str(out_path))
    print(f"Saved t-SNE plot to {out_path}")

if __name__ == "__main__":
    main()
