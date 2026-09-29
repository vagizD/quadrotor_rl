import sys
from pathlib import Path

# Add src to sys.path so we can import packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from research.pipeline.control.residual.models.models import NeuralFlyModel

def main():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
        
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "px4_hover.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        
    res_cfg = config.get("residual_model", {})
    
    tsne_data_path = Path(__file__).resolve().parent / "runs" / "tsne_data.npz"
    if not tsne_data_path.exists():
        print(f"Data {tsne_data_path} not found. Run train_residual_model.py first.")
        return
        
    data = np.load(tsne_data_path)
    winds = data["winds"]
    
    # We want to randomly subsample to 2000 points if it's too large
    N = len(winds)
    if N > 2000:
        idx = np.random.choice(N, 2000, replace=False)
        winds = winds[idx]
        
    model = NeuralFlyModel(
        state_dim=13, 
        action_dim=4, 
        wind_dim=3,
        num_basis=res_cfg.get("num_basis", 16)
    )
    
    model_save_path_str = res_cfg.get("model_path", "models/residual_model.pt")
    model_save_path = Path(__file__).resolve().parent.parent.parent / model_save_path_str
    
    if not model_save_path.exists():
        print("Model not trained yet.")
        return
        
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        winds_tensor = torch.tensor(winds, dtype=torch.float32)
        latents = model.wind_encoder(winds_tensor).numpy()
        
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)
    
    # Compute wind speeds to use as colors
    wind_speeds = np.linalg.norm(winds, axis=-1)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=wind_speeds, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Wind Speed (m/s)")
    plt.title("t-SNE visualization of Wind Latent Representation (a(w))")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    out_path = Path(__file__).resolve().parent / "runs" / "tsne_latents.png"
    plt.savefig(out_path)
    print(f"Saved t-SNE plot to {out_path}")
    
if __name__ == "__main__":
    main()
