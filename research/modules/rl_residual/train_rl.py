from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import torch

from research.pipeline.control.residual.env import ResidualLearningEnv

def main():
    # Make a vectorized environment
    env = make_vec_env(lambda: ResidualLearningEnv(history_len=10, max_steps=500), n_envs=4)
    eval_env = make_vec_env(lambda: ResidualLearningEnv(history_len=10, max_steps=500), n_envs=1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training RL on device: {device}")
    
    tb_log = "research/modules/rl_residual/tensorboard_logs"
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=tb_log,
        device=device
    )
    
    models_dir = Path("research/modules/rl_residual/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=str(models_dir),
        log_path=str(models_dir),
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train
    print("Starting RL training for residual prediction...")
    # Train for 30k timesteps
    model.learn(total_timesteps=30_000, callback=eval_callback)
    
    # Save the final model
    save_path = models_dir / "rl_residual_model.zip"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
