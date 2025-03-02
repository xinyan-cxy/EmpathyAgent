import random  
import json 
import os 
import csv
import time
from tqdm import tqdm
from pathlib import Path  
import sys   
from gpt import GPT
import argparse
from .overlap import Overlap, TF_IDF, LCS
from .NLG_metric import BERTScore
from .reference_free_metrics.api_eval import EmpathyEvaluator
from .reference_free_metrics.legality import LegalityChecker
from .reference_free_metrics.scorer import EmpathyScorer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="Choose between 'gpt-4o','gpt-4-turbo','gpt-4-vision-preview'",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="empathetic_action",
        help="Choose between 'scenario_understanding','empathetic_planning','empathetic_action'",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../dataset/testset_100.json",
        help="The path of the test file",
    )
    parser.add_argument(
        "--reference_free_eval",
        action="store_true",
        help="Enable reference-free evaluation (default: False)",
    )
    args = parser.parse_args()
    return args

def load_existing_indices(csv_file_path):
    existing_indices = set()
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_indices.add(int(row['data_idx']))
    return existing_indices

def inference(model, task, model_name = "", empathy_scenario_data_path = "", character_data_path = "", video_path = "", script_path = ""):
    response_list = []
    empathy_scenario_data = json.load(open(empathy_scenario_data_path, 'r', encoding='utf-8')) 
    character_data = json.load(open(character_data_path,'r'))
    
    os.makedirs(os.path.join("output", task), exist_ok=True)
    csv_file_path = os.path.join("output", task, f"{model_name}.csv")
    existing_indices = load_existing_indices(csv_file_path)
    
    if task == "scenario_understanding":
        with open("./prompt/prompt_video_l1.txt", "r", encoding="utf-8") as file:
            prompt = file.read().strip()
    elif task == "empathetic_planning":
        with open("./prompt/prompt_video_l2.txt", "r", encoding="utf-8") as file:
            prompt = file.read().strip()
    elif task == "empathetic_action":   
        with open("./prompt/prompt_video_l3.txt", "r", encoding="utf-8") as file:
            prompt = file.read().strip()
    
    with open(csv_file_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not existing_indices:  # Write the header only if the file is empty or does not exist
            writer.writerow(["data_idx", "response"])
        for idx, data in tqdm(enumerate(empathy_scenario_data)):
            if idx in existing_indices:
                continue
            character_id, dialogue, action_id = data["character_id"], data["dialogue"], data["action_id"]
            if video_path:
                video_or_script_input = os.path.join(video_path, f"{action_id}.mp4")
            else:
                video_or_script_input = os.path.join(script_path, f"{action_id}/script/0")
            character_info = character_data[str(character_id)]
            print(video_or_script_input)
            input_prompt = prompt.format(character_info = character_info, dialogue = dialogue)
            try:
                response = model.generate(video_or_script_input, input_prompt)
                response_list.append(response)
                writer.writerow([idx, response])  
            except Exception as e:
                if 'Quota exceeded' in str(e):
                    print("Man, what can I say? I'm out of quota. Exiting now.")
                    raise e  
                else:
                    raise e   
    print(f"Inference Done for {model_name}!")


if __name__ == "__main__":
    args = parse_args()
    
    # inference
    if args.model_name in ["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview"]:
        gpt = GPT(model_name = args.model_name)
        inference(gpt,
                  args.task,
                  args.model_name,
                  empathy_scenario_data_path = args.test_file, 
                  character_data_path = "../dataset/character.json", 
                  script_path = "../dataset/scripts")
    else:
        print("Model name is wrong!")
        sys.exit(1)
        
    # evaluation
    csv_file = f"./output/{args.task}/{args.model_name}.csv"
    response_dict = {}
    with open(csv_file, 'r', newline='', encoding='latin-1') as file:
        reader = csv.DictReader(file)
        for row in reader:
            response = row['response']
            idx = row['data_idx']
            if response is not None and response.strip():
                response_dict[f'{idx}'] = response
                
    if args.task == "empathetic_action":
        if args.reference_free_eval:
            reference_free_metric_output_file = f"../output/{args.task}/reference_free_metrics_{args.model_name}.json"
            evaluator = EmpathyEvaluator(test_json_file=args.test_file, level=3)
            evaluator.evaluate(csv_file=csv_file, output_file = reference_free_metric_output_file)
            # legality
            checker = LegalityChecker(
                csv_file=csv_file,
                output_file=reference_free_metric_output_file
            )
            checker.process(verbose=True)
            # print score
            scorer = EmpathyScorer(result_path=f"../output/{args.task}/reference_free_metrics_{args.model_name}_legality.json", level=3)
            results = scorer.run()
            print("Results of reference free metrics:")
            scorer.print_results()
            
        print("Results of reference based metrics:")
        overlap = Overlap()
        overlap.score(response_dict, args.test_file)
        lcs = LCS()
        lcs.score(response_dict, args.test_file)
        tf_idf = TF_IDF()
        tf_idf.score(response_dict, args.test_file)
        
    elif args.task == "empathetic_planning" or args.task == "scenario_understanding":
        if args.reference_free_eval:
            reference_free_metric_output_file = f"../output/{args.task}/reference_free_metrics_{args.model_name}.csv"
            evaluator = EmpathyEvaluator(test_json_file=args.test_file, level = 2 if args.task == "empathetic_planning" else 1)
            evaluator.evaluate(csv_file=csv_file, output_file=reference_free_metric_output_file)
            # print score
            scorer = EmpathyScorer(result_path=reference_free_metric_output_file, level = 2 if args.task == "empathetic_planning" else 1)
            results = scorer.run()
            print("Results of reference free metrics:")
            scorer.print_results()
            
        print("Results of reference based metrics:")
        bert_score = BERTScore(model_dir = "google-bert/bert-base-uncased")
        bert_score.score(response_dict, args.test_file, test_level=args.task)

