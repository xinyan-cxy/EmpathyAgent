import random  
import json 
import os 
import csv
import time
from tqdm import tqdm
from overlap import Overlap 
from overlap import TF_IDF 
from overlap import LCS 
# from gemini import Gemini
from gpt import GPT
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve() / "reward_model_metric"))
from rewardmodel import LlaMaRewardModel as llamarm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        default="LCS",
        help="Choose between 'RM','overlap','tfidf','LCS'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="Choose between 'gpt-4o','gpt-4-turbo','gpt-4-vision-preview','gemini-1.0-pro-vision-001','gpt-3.5-turbo-0125'",
    )
    parser.add_argument(
        "--testset_path",
        type=str,
        default="./dataset_scale_up/testset_100.json",
        help="path of testset file",
    )
    args = parser.parse_args()
    return args

def llama_reward_model_eval(inference_response_list, source_data_path):
    """
    Args:
        - input_response_list: (_type_): Inference Response List returned from inference function. 
    Note : The order for Inference Result must be alligned with the source_data_path in rank_10k_gpt4_new.json 
    """
    model_name_or_path = "./OpenRLHF/examples/scripts/ckpt/7b_llama"
    llama_rm = llamarm(
        model_name_or_path,
        source_data_path = source_data_path,
        source_character_path = "./dataset_scale_up/character.json", 
        use_flash_attention_2=False,
        bf_16=True,
        lora_rank=0,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0,
        #ds_config = json.load(open(ds_config,'r')),
        ds_config=None,
        init_value_head=False,
    )
    reward_diff_list, human_reward_list, model_generated_reward_list = llama_rm.score(inference_response_list)
    return reward_diff_list, human_reward_list, model_generated_reward_list

def llama_reward_average_score(reward_diff_list):
    if not reward_diff_list: 
        return 0
    total_score = sum(reward_diff_list)  
    average_score = total_score / len(reward_diff_list)  
    return average_score

if __name__ == "__main__":
    args = parse_args()
    response_dict = {}
    with open(args.model_name + ".csv", 'r', newline='', encoding='latin-1') as file:
        reader = csv.DictReader(file)
        for row in reader:
            response = row['response']
            idx = row['data_idx']
            if response is not None and response.strip():
                response_dict[f'{idx}'] = response
    # print(response_list)
    response_list = []
    with open(args.model_name + ".csv", 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            response_list.append(row['response'])
            
    if args.metric == "RM":
        reward_diff_list, human_reward_list, model_generated_reward_list = llama_reward_model_eval(response_list, args.testset_path)
        score = llama_reward_average_score(reward_diff_list)
        print("Average RM score:", 1+score)
        
    elif args.metric == "overlap":
        overlap = Overlap()
        overlap.score(response_dict, args.testset_path)
        
    elif args.metric == "tfidf":
        tf_idf = TF_IDF()
        tf_idf.score(response_dict, args.testset_path)
        
    elif args.metric == "LCS":
        lcs = LCS()
        lcs.score(response_dict, args.testset_path)
    
    else:
        print("Metric name is wrong!")