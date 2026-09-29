import os
import glob
import shutil

def makedirs(path):
    os.makedirs(path, exist_ok=True)

def move(src, dst):
    if os.path.exists(src):
        # if dst is a directory, it moves it inside. If dst ends with a filename, it renames.
        if os.path.isdir(dst) and not dst.endswith('/'):
             dst = dst + '/'
        shutil.move(src, dst)
        print(f"Moved {src} to {dst}")
    else:
        print(f"Warning: {src} does not exist.")

def replace_in_files(replacements):
    for root, _, files in os.walk('.'):
        if '.venv' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.toml'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                changed = False
                for old_str, new_str in replacements.items():
                    if old_str in content:
                        content = content.replace(old_str, new_str)
                        changed = True
                
                if changed:
                    with open(filepath, 'w') as f:
                        f.write(content)
                    print(f"Updated {filepath}")

# 1. Create structure
makedirs("projects/px4_baseline")
makedirs("projects/neural_fly_adaptive")
makedirs("projects/rl_hover")
makedirs("src/core/physics")
makedirs("src/core/math")
makedirs("src/controllers/px4")
makedirs("src/controllers/adaptive")
makedirs("src/planning/generators")
makedirs("src/planning/collision")

makedirs("tests/core/physics")
makedirs("tests/core/math")
makedirs("tests/controllers")

# 2. Move files
# Geometry to core/math
if os.path.exists("src/common/geometry"):
    for item in os.listdir("src/common/geometry"):
        move(f"src/common/geometry/{item}", f"src/core/math/{item}")
    os.rmdir("src/common/geometry")
if os.path.exists("tests/common/geometry"):
    for item in os.listdir("tests/common/geometry"):
        move(f"tests/common/geometry/{item}", f"tests/core/math/{item}")
    shutil.rmtree("tests/common/geometry")

# Controllers
if os.path.exists("src/robots/quadrotor/controllers"):
    move("src/robots/quadrotor/controllers/px4.py", "src/controllers/px4/px4.py")
    move("src/robots/quadrotor/controllers/residual_px4.py", "src/controllers/adaptive/residual_px4.py")
    move("src/robots/quadrotor/controllers/__init__.py", "src/controllers/__init__.py")
    shutil.rmtree("src/robots/quadrotor/controllers")

# Scripts -> Projects
move("scripts/run_px4_scenarios.py", "projects/px4_baseline/run_scenarios.py")
move("scripts/diagnose_physics.py", "projects/px4_baseline/diagnose_physics.py")
move("scripts/train_residual_model.py", "projects/neural_fly_adaptive/train_residual.py")
move("scripts/evaluate_residual.py", "projects/neural_fly_adaptive/evaluate_residual.py")
move("scripts/generate_dr_data.py", "projects/neural_fly_adaptive/generate_dr_data.py")
move("scripts/tsne_latent.py", "projects/neural_fly_adaptive/tsne_latent.py")

# Experiments -> learning/experiments
if os.path.exists("experiments"):
    move("experiments", "src/learning/experiments")

# 3. Search and replace imports
replacements = {
    "from geometry.math": "from geometry.math",
    "import core.math": "import core.math",
    "from software.pipeline.control.px4.px4": "from software.pipeline.control.px4.px4",
    "from software.pipeline.control.adaptive.residual_px4": "from software.pipeline.control.adaptive.residual_px4",
    "from research.pipeline.control.residual.models.experiments.": "from research.pipeline.control.residual.models.experiments.",
    "import research.pipeline.control.residual.models.experiments.": "import research.pipeline.control.residual.models.experiments.",
}
replace_in_files(replacements)
print("Done!")
