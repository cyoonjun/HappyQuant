import os
import yaml
import re
import pandas as pd

def extract_perplexity_as_dataframe():
    root_dir = './checkpoints/ptq/'
    data = []

    for init_name_dir in os.listdir(root_dir):
        init_path = os.path.join(root_dir, init_name_dir)
        if not os.path.isdir(init_path):
            continue
        init_name = init_name_dir

        for scale_mode_dir in os.listdir(init_path):
            scale_path = os.path.join(init_path, scale_mode_dir)
            if not os.path.isdir(scale_path):
                continue
            scale_mode = scale_mode_dir

            for model_name_dir in os.listdir(scale_path):
                model_path = os.path.join(scale_path, model_name_dir)
                if not os.path.isdir(model_path):
                    continue
                model_name = model_name_dir

                # We iterate through cal_set_cal_num, but we don't explicitly need these values
                for cal_dir in os.listdir(model_path):
                    cal_path = os.path.join(model_path, cal_dir)
                    if not os.path.isdir(cal_path):
                        continue

                    for q_dir in os.listdir(cal_path):
                        match_q = re.match(r'([^_]+)_(\d+)', q_dir)
                        if not match_q:
                            continue
                        q_name, q_bit_str = match_q.groups()
                        q_bit = int(q_bit_str)
                        q_path = os.path.join(cal_path, q_dir)
                        if not os.path.isdir(q_path):
                            continue

                        for lr_itr_dir in os.listdir(q_path):
                            match_lr_itr = re.match(r'(\d+)_(\d+)', lr_itr_dir)
                            if not match_lr_itr:
                                continue
                            lr_rank_str, itr_str = match_lr_itr.groups()
                            itr = int(itr_str)

                            if q_bit == 3 and lr_rank_str == '64' and itr==1:
                              
                                hella = None  
                                wino_res = None
                                bolq_res = None
                                mmlu = None
                                bbh = None
                                
                                task_file_path = os.path.join(q_path, lr_itr_dir, 'lm_eval_results.yaml')
                                if os.path.exists(task_file_path):
                                    try:
                                        with open(task_file_path, 'r') as f:
                                            task_data = yaml.load(f, Loader=yaml.UnsafeLoader)
                                            res = task_data['results']
                                            ########### extract ###########
                                            bolq_res       = res.get('boolq', {}).get('acc,none', None)
                                            mmlu           = res.get('mmlu', {}).get('acc,none', None)
                                            wino_res       = res.get('winogrande', {}).get('acc,none', None)
                                            hella          = res.get('hellaswag',{}).get('acc_norm,none', None)

                                            ########### BBH ###########
                                            bbh = [
                                                res[key].get('acc_norm,none', None)
                                                for key in res
                                                if key.startswith('leaderboard_bbh') and 'acc_norm,none' in res[key]
                                            ]
                                            bbh = sum(bbh) / len(bbh) if bbh else None

                                    except FileNotFoundError:
                                        print(f"Warning: YAML file not found at {task_file_path}")
                                    except yaml.YAMLError as e:
                                        print(f"Error reading YAML file at {task_file_path}: {e}")

                                yaml_file_path = os.path.join(q_path, lr_itr_dir, 'perplexity_results.yaml')
                                if os.path.exists(yaml_file_path):
                                    try:
                                        with open(yaml_file_path, 'r') as f:
                                            yaml_data = yaml.safe_load(f)
                                            perplexity = None
                                            perplexity = yaml_data.get('perplexity')
                                            if perplexity is not None:
                                                data.append({
                                                    'model_name': model_name,
                                                    'q_name': q_name,
                                                    'q_bit': q_bit,
                                                    'init_name': init_name,
                                                    'itr': itr,
                                                    'scale_mode': scale_mode,   
                                                    'hella': str(round(hella*100, 2))+',' if hella is not None else 'N/A,',
                                                    'wino_res': str(round(wino_res*100, 2))+',' if wino_res is not None else 'N/A,',
                                                    'boolq_res': str(round(bolq_res*100, 2))+',' if bolq_res is not None else 'N/A,',
                                                    'mmlu_res': str(round(mmlu*100, 2))+',' if mmlu is not None else 'N/A,',
                                                    'bbh_res': str(round(bbh*100, 2)) if bbh is not None else 'N/A,',
                                                })
                                    except FileNotFoundError:
                                        print(f"Warning: YAML file not found at {yaml_file_path}")
                                    except yaml.YAMLError as e:
                                        print(f"Error reading YAML file at {yaml_file_path}: {e}")

    df = pd.DataFrame(data)
    desired_order = ['model_name', 'q_name', 'q_bit', 'init_name', 'itr', 'scale_mode', 'hella', 'wino_res', 'boolq_res', 'mmlu_res', 'bbh_res']
    df = df.reindex(columns=desired_order)
    return df

if __name__ == "__main__":
    perplexity_df = extract_perplexity_as_dataframe()
    if not perplexity_df.empty:
        print(perplexity_df)
        os.makedirs('perplexity_results', exist_ok=True)
        excel_filename = './perplexity_results/ppl_table.xlsx'
        perplexity_df.to_excel(excel_filename, index=False)
        print(f"\nPerplexity results saved to {excel_filename}")
    else:
        print("No perplexity results found.")