import os
import yaml
import csv
import math

def formatted_result(dir: str) -> None: 
    """
    Read the contents of eval.txt to extract the required values, 
    and then format the results into a CSV file.
    """
    csv_file = 'formatted_result.csv'
    csv_columns = ['DirName', 'Reward', 'PE']
    file_handler = open(csv_file, f"w+", newline="\n")
    writer = csv.writer(file_handler)
    writer.writerow(csv_columns)  # Write header

    def custom_key(x):
        order1 = x.split('_')[0].split('v')[0]
        if 'PPO' in x:
            order2 = 1
        elif 'ACKTR' in x:
            order2 = 2
        order3= int(x.split('-h')[1].split('-')[0])
        order4 = int(x.split('-c')[1].split('-')[0])
        order5 = int(x.split('-n')[1].split('-')[0])
        order6 = int(x.split('-b')[1].split('-')[0])
        if '-M' in x:
            order8 = 1
            if '-Me' in x: 
                order7 = int(math.pow(10, int(x.split('-Me')[1])))
            else:
                order7 = int(x.split('-M')[1]) 
        elif '-R' in x:
            order8 = 2
            if '-Re' in x: 
                order7 = int(math.pow(10, int(x.split('-Re')[1])))
            else:
                order7 = int(x.split('-R')[1]) 
        order9= int(x.split('-k')[1].split('-')[0])
        if 'rA' in x:
            order10 = 1
        elif 'rC' in x:
            order10 = 2
        return (order1, order2, order3, order4, order5, order6, order7, order8, order9, order10)

    sorted_dirs = sorted(os.listdir(dir), key=custom_key)
    for dir_name in sorted_dirs:
        print(dir_name)
        eval_file_path = os.path.join(dir, dir_name, 'eval.txt')
        if os.path.isfile(eval_file_path):
            with open(eval_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'reward' in line:
                        reward = line.split(': ')[-1]
                    if 'PE' in line:
                        PE = line.split(': ')[-1]
            result = [dir_name.split('2DBpp-')[-1], reward, PE]
            writer.writerow(result)


def update_yaml_files(dir):
    """
    update the YAML files according to the filename
    """
    for filename in os.listdir(dir):
        if filename.endswith(".yaml"):  # 確保處理的是YAML文件
            filepath = os.path.join(dir, filename)
            with open(filepath, 'r') as file:
                data = yaml.load(file, Loader=yaml.UnsafeLoader)  # 讀取YAML文件

            if "v" in filename:
                if "v1" in filename:
                    data['env_id'] = '2DBpp-v1'
                    data['total_timesteps'] = 3000000
                    data['env_kwargs']['items_per_bin'] = 15
                    data['policy_kwargs']['network'] = "CnnMlpNetwork1"
                elif "v2" in filename:
                    data['env_id'] = '2DBpp-v2'
                    data['total_timesteps'] = 6000000
                    data['env_kwargs']['items_per_bin'] = 20
                    data['policy_kwargs']['network'] = "CnnMlpNetwork2"
                elif "v3" in filename:
                    data['env_id'] = '2DBpp-v3'
                    data['total_timesteps'] = 12500000
                    data['env_kwargs']['items_per_bin'] = 25
                    data['policy_kwargs']['network'] = "CnnMlpNetwork3"
                elif "v4" in filename:
                    data['env_id'] = '2DBpp-v4'
                    data['total_timesteps'] = 12500000
                    data['env_kwargs']['items_per_bin'] = 25
                    data['policy_kwargs']['network'] = "CnnMlpNetwork4"
            else:
                print("Lack of environment version!\n")
                return

            if "-h" in filename:
                hidden = int(filename.split('-h')[1].split('-')[0])
                data['policy_kwargs']['network_kwargs']['hidden_dim'] = hidden
            else:
                print("Lack of hidden dimension!\n")
                return
                
            if "-c" in filename:
                clip_range = float(filename.split('-c')[1].split('-')[0]) / 10
                data['PPO_kwargs']['clip_range'] = clip_range
            else:
                print("Lack of clip range!\n")
                return
            
            if "-n" in filename:
                n_steps = int(filename.split('-n')[1].split('-')[0])
                data['PPO_kwargs']['n_steps'] = n_steps
            else:
                print("Lack of n_steps!\n")
                return
            
            if "-b" in filename:
                batch_size = int(filename.split('-b')[1].split('-')[0])
                data['PPO_kwargs']['batch_size'] = batch_size
            else:
                print("Lack of batch_size!\n")
                return

            if '-R' in filename:
                if '-Re' in filename:
                    mask_coef = int(math.pow(10, int(filename.split('-Re')[1].split('-')[0])))
                else:
                    mask_coef = int(filename.split('-R')[1].split('-')[0])
                data['policy_kwargs']['dist_kwargs']['mask_strategy'] = 'replace'
                data['policy_kwargs']['dist_kwargs']['mask_replace_coef'] = -mask_coef
            elif '-M' in filename:
                if '-Me' in filename:
                    mask_coef = int(math.pow(10, int(filename.split('-Me')[1].split('-')[0])))
                else:
                    mask_coef = int(filename.split('-M')[1].split('-')[0])
                data['policy_kwargs']['dist_kwargs']['mask_strategy'] = 'minus'
                data['policy_kwargs']['dist_kwargs']['mask_minus_coef'] = mask_coef
            else:
                print("Lack of mask strategy!\n")
                return
            
            if "-k" in filename:
                n_epochs = int(filename.split('-k')[1].split('-')[0])
                data['PPO_kwargs']['n_epochs'] = n_epochs
            else:
                print("Lack of n_epochs!\n")
                return

            if '-r' in filename:
                if '-rA' in filename:
                    data['env_kwargs']['reward_type'] = 'area'
                elif '-rC' in filename:
                    data['env_kwargs']['reward_type'] = 'compactness'
            else:
                print("Lack of reward type!\n")
                return
        
            if '-P' in filename:
                data['policy_kwargs']['mask_type'] = 'predict'
            else:
                data['policy_kwargs']['mask_type'] = 'truth'

            # update the YAML file
            with open(filepath, 'w') as file:
                yaml.dump(data, file, sort_keys=False)


def copy_file1(dir: str) -> None: 
    import shutil
    source_file = "main/v4_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml"
    destination_files = [
        "v4_PPO-h1600-c02-n64-b32-R0-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-R7-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-R30-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-R50-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-R100-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-R500-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-Re3-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-Re4-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-Re5-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-Re6-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-Re7-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b32-Re8-k1-rA.yaml",
    ]
    for destination_file in destination_files:
        shutil.copy(os.path.join(dir, source_file), os.path.join(dir, destination_file))


def copy_file2(dir: str) -> None: 
    import shutil
    source_file = "main/v4_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml"
    destination_files = [
        "v4_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b64-R15-k5-rA.yaml",
        "v4_PPO-h1600-c02-n128-b32-R15-k1-rA.yaml",
        "v4_PPO-h1600-c02-n128-b64-R15-k5-rA.yaml",
        "v4_PPO-h1600-c04-n64-b32-R15-k1-rA.yaml",
        "v4_PPO-h1600-c04-n64-b64-R15-k5-rA.yaml",
        "v4_PPO-h1600-c04-n128-b32-R15-k1-rA.yaml",
        "v4_PPO-h1600-c04-n128-b64-R15-k5-rA.yaml",
        "v4_PPO-h3200-c02-n64-b32-R15-k1-rA.yaml",
        "v4_PPO-h3200-c02-n64-b64-R15-k5-rA.yaml",
        "v4_PPO-h3200-c02-n128-b32-R15-k1-rA.yaml",
        "v4_PPO-h3200-c02-n128-b64-R15-k5-rA.yaml",
        "v4_PPO-h3200-c04-n64-b32-R15-k1-rA.yaml",
        "v4_PPO-h3200-c04-n64-b64-R15-k5-rA.yaml",
        "v4_PPO-h3200-c04-n128-b32-R15-k1-rA.yaml",
        "v4_PPO-h3200-c04-n128-b64-R15-k5-rA.yaml",
    ]
    for destination_file in destination_files:
        shutil.copy(os.path.join(dir, source_file), os.path.join(dir, destination_file))

if __name__ == "__main__":
    # $ tensorboard --logdir backup/diff_mask
    # formatted_result("./logs/ppo")
    # copy_file1("./settings")
    copy_file2("./settings")
    update_yaml_files("./settings")