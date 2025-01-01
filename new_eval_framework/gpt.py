from openai import OpenAI
import os
import json
import httpx
import re
import time
from tqdm import tqdm
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

LEVEL = 2

def extract_reasoning(text):
    match = re.search(r'Reasoning(.*?)(?=\s*Score)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def find_score(text):
    parts = text.split("Score")
    if len(parts) > 1:
        score_part = parts[1].strip() 
        match = re.search(r'\b(10|[0-9](\.[0-9]+)?|[0-9]\.[0-9]+)\b', score_part)
        if match:
            return float(match.group(1))
    return None

def load_results(output_file):
    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            return json.load(file)
    return {}

def save_results(result_dict, output_file):
    with open(output_file, "w") as file:
        json.dump(result_dict, file, indent=4)

def process_index(idx, response_dict, test_data, character_data, action_data, metrics, output_file):
    if response_dict.get(f'{idx}'):
        response = response_dict[f'{idx}']
        character_id = test_data[idx]["character_id"]
        action_id = test_data[idx]["action_id"]
        dialogue = test_data[idx]["dialogue"]
        scenario = test_data[idx]["scenario"]

        character = character_data[str(character_id)]
        action = action_data[str(action_id)]

        result_item = {}
        for metric in metrics:
            with open(f"./metrics/{metric}.txt", "r", encoding="utf-8") as file:
                prompt_metric = file.read().strip()

            if LEVEL == 1:
                input_prompt = PROMPT_PREFIX_L1.format(character=character, action=action, dialogue=dialogue, scenario=scenario, response=response) + prompt_metric
            else:
                input_prompt = PROMPT_PREFIX.format(character=character, action=action, dialogue=dialogue, scenario=scenario, response=response) + prompt_metric
            # print(input_prompt)
            
            client = OpenAI(
                base_url="", 
                api_key="",
                http_client=httpx.Client(
                    base_url="",
                    follow_redirects=True,
                ),
            )

            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                temperature=0,
                messages=[
                    {"role": "user", "content": input_prompt}
                ],
                max_tokens=1024
            )
            result = completion.choices[0].message.content
            
            if find_score(result) and extract_reasoning(result) is not None:
                result_item[f"{metric}"] = {
                    "score": find_score(result),
                    "reason": extract_reasoning(result)
                }
            else:
                result_item[f"{metric}"] = {
                    "score": None,
                    "reason": result
                }
                
        result_dict = load_results(output_file)  # Load existing results
        result_dict[f"{idx}"] = result_item
        save_results(result_dict, output_file)  # Save results after processing

# Constants
PROMPT_PREFIX = """ 
There is a character at home and the character's background information, scenario, action and dialogue are provided.
There is also an empathetic robot which can make empathetic responses to the character.

The interaction between the character and the robot is as follows:
Character information: {character}
Scenario: {scenario}
Character action: {action}
Character dialogue: {dialogue}
Robot response:  {response}

"""
PROMPT_PREFIX_L1 = """ 
There is a character at home and the character's background information, scenario, action and dialogue are provided.
There is also an empathetic robot which can infer the scenario without relying on the provided ground truth scenario.

The information of the character and the response of the robot is as follows:
Character information: {character}
Scenario: {scenario}
Character action: {action}
Character dialogue: {dialogue}
Robot response: {response}

"""

csv_file = './empathy_robotic_data/baseline/Llava_scenario_gt_l2.csv'
test_json_file = "./dataset_scale_up/testset_100.json" 
output_file = "./results/Llava_scenario_gt_l2.json"

# Load data
response_dict = {}
with open(csv_file, 'r', newline='', encoding='latin-1') as file:
    reader = csv.DictReader(file)
    for row in reader:
        response = row['response']
        idx = row['data_idx']
        if response is not None and response.strip():
            response_dict[f'{idx}'] = response
            
with open(test_json_file, 'r', encoding="latin1") as f:
    test_data = json.load(f)
    
character_data = json.load(open("./dataset_scale_up/character.json",'r'))
action_data = json.load(open("./dataset_scale_up/action_list.json",'r'))

if LEVEL == 1:
    metrics = ["association", "coherence_l1", "emotional_com_l1", "individual"]
else:
    metrics = ["adaptability", "association", "coherence", "emotion_reg", "emotional_com_l23", "helpfulness", "individual"]

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_index, idx, response_dict, test_data, character_data, action_data, metrics, output_file): idx for idx in range(1,100)}
    
    for future in tqdm(as_completed(futures)):
        idx = futures[future]
        try:
            future.result()  # Ensure any exceptions are raised
        except Exception as e:
            print(f"Error processing index {idx}: {e}")

# Final write to ensure all data is saved
# No need for this since results are saved in each process.
