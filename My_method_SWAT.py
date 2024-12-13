import yaml
import subprocess

model = 'My_method'
dataset = 'SWAT'
config_path = f'./config/{model}.yaml'
main_path = './main.py'

# Step 1: Update `main.py` to dynamically load the config
with open(main_path, 'r') as f:
    lines = f.readlines()

lines[23] = f"with open('./config/{model}.yaml', 'r') as f:\n"

with open(main_path, 'w') as f:
    f.writelines(lines)

# Step 2: Modify the configuration file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['model'] = model
config['dataset'] = dataset

# Step 3: Run multiple experiments with different seeds
loss_types=['mse','dynamic_loss']
"""首先使用mse作为损失函数跑"""
for loss_type in loss_types:
    config[f'{dataset}'][f'{loss_type}']=loss_type
    for i in range(5):  # Run 5 iterations
        config['seed'] = i

        # Save the updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Execute the script
        result = subprocess.run(['python', './main.py'], capture_output=True, text=True)

        # Log the result if needed
        print(f"Run {i}, Seed {i}:")
        print(result.stdout)
        print(result.stderr)
