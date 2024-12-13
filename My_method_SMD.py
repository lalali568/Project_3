import yaml
import subprocess
from tqdm import tqdm

model = 'My_method'
dataset = 'SMD'
config_path = f'./config/{model}.yaml'
main_path = './main.py'
repeat_times=1 #表示，要跑多少次重复实验

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
config['plot_flag']=0# 跑很多的次的情况下，就不绘图了，浪费时间

# SMD has multiple files
files_names = [
    '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8',
    '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9',
    '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9',
    '3-10', '3-11'
]

# Step 3: Run multiple experiments with different seeds
dynamic_picks=[0,1]#0代表False,1代表True
for flag in dynamic_picks:
    config[f'{dataset}']['dynamic_pick']=flag
    for i in tqdm(range(repeat_times)):
        config['seed'] = i

        for files_name in files_names:
            # Update test data and label paths for each file
            config[f'{dataset}']['test_data_path'] = f'data/SMD/test/csv_files/machine-{files_name}.csv'
            config[f'{dataset}']['test_label_path'] = f'data/SMD/test_label/csv_files/machine-{files_name}.csv'

            # Save the updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            # Execute the script
            result = subprocess.run(['python', './main.py'], capture_output=True, text=True)

            # Log the result if needed
            print(f"Run {i}, File {files_name}, Seed {i}:")
            print(result.stdout)
            print(result.stderr)
