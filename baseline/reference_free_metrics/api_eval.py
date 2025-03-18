from openai import OpenAI
import os
import json
import httpx
import re
import time
from tqdm import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

class EmpathyEvaluator:
    def __init__(self, test_json_file, level=2):
        self.test_json_file = test_json_file
        self.LEVEL = level
        
        # Constants
        self.PROMPT_PREFIX = """ 
There is a character at home and the character's background information, scenario, action and dialogue are provided.
There is also an empathetic robot which can make empathetic responses to the character.

The interaction between the character and the robot is as follows:
Character information: {character}
Scenario: {scenario}
Character action: {action}
Character dialogue: {dialogue}
Robot response:  {response}

"""
        self.PROMPT_PREFIX_L1 = """ 
There is a character at home and the character's background information, scenario, action and dialogue are provided.
There is also an empathetic robot which can infer the scenario without relying on the provided ground truth scenario.

The information of the character and the response of the robot is as follows:
Character information: {character}
Scenario: {scenario}
Character action: {action}
Character dialogue: {dialogue}
Robot response: {response}

"""
        # Set metrics based on level
        if self.LEVEL == 1:
            self.metrics = ["association", "coherence_l1", "emotional_com_l1", "individual"]
        else:
            self.metrics = ["adaptability", "association", "coherence", "emotion_reg", "emotional_com_l23", "helpfulness", "individual"]
            
        # Load datasets
        self.character_data = json.load(open("../dataset/character.json",'r'))
        self.action_data = json.load(open("../dataset/action_list.json",'r'))
        
        with open(self.test_json_file, 'r', encoding="latin1") as f:
            self.test_data = json.load(f)

    def extract_reasoning(self, text):
        match = re.search(r'Reasoning(.*?)(?=\s*Score)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def find_score(self, text):
        parts = text.split("Score")
        if len(parts) > 1:
            score_part = parts[1].strip() 
            match = re.search(r'\b(10|[0-9](\.[0-9]+)?|[0-9]\.[0-9]+)\b', score_part)
            if match:
                return float(match.group(1))
        return None

    def load_results(self, output_file):
        if os.path.exists(output_file):
            with open(output_file, "r") as file:
                return json.load(file)
        return {}

    def save_results(self, result_dict, output_file):
        with open(output_file, "w") as file:
            json.dump(result_dict, file, indent=4)

    def process_index(self, idx, response_dict, output_file):
        if response_dict.get(f'{idx}'):
            response = response_dict[f'{idx}']
            character_id = self.test_data[idx]["character_id"]
            action_id = self.test_data[idx]["action_id"]
            dialogue = self.test_data[idx]["dialogue"]
            scenario = self.test_data[idx]["scenario"]

            character = self.character_data[str(character_id)]
            action = self.action_data[str(action_id)]

            result_item = {}
            for metric in self.metrics:
                with open(f"reference_free_metrics/metrics/{metric}.txt", "r", encoding="utf-8") as file:
                    prompt_metric = file.read().strip()

                if self.LEVEL == 1:
                    input_prompt = self.PROMPT_PREFIX_L1.format(character=character, action=action, dialogue=dialogue, scenario=scenario, response=response) + prompt_metric
                else:
                    input_prompt = self.PROMPT_PREFIX.format(character=character, action=action, dialogue=dialogue, scenario=scenario, response=response) + prompt_metric
                
                client = OpenAI(
                    base_url=os.environ.get("OPENAI_API_BASE", ""), 
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                    http_client=httpx.Client(
                        base_url=os.environ.get("OPENAI_API_BASE", ""), 
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
                
                if self.find_score(result) and self.extract_reasoning(result) is not None:
                    result_item[f"{metric}"] = {
                        "score": self.find_score(result),
                        "reason": self.extract_reasoning(result)
                    }
                else:
                    result_item[f"{metric}"] = {
                        "score": None,
                        "reason": result
                    }
                    
            result_dict = self.load_results(output_file)  # Load existing results
            result_dict[f"{idx}"] = result_item
            self.save_results(result_dict, output_file)  # Save results after processing
            # print(f"Processed index {idx}")

    def load_responses(self, csv_file):
        response_dict = {}
        with open(csv_file, 'r', newline='', encoding='latin-1') as file:
            reader = csv.DictReader(file)
            for row in reader:
                response = row['response']
                idx = row['data_idx']
                if response is not None and response.strip():
                    response_dict[f'{idx}'] = response
        return response_dict

    def evaluate(self, csv_file, output_file, max_workers=10):
        response_dict = self.load_responses(csv_file)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:  
            futures = {
                executor.submit(self.process_index, idx, response_dict, output_file): idx 
                for idx in range(len(self.test_data))
            }

            for future in tqdm(as_completed(futures)):
                idx = futures[future]
                try:
                    future.result()  
                except Exception as e:
                    print(f"Error processing index {idx}: {e}")


if __name__ == "__main__":
    test_json_file = "../dataset/testset_100.json"
    LEVEL = 2
    
    evaluator = EmpathyEvaluator(test_json_file=test_json_file, level=LEVEL)
    
    csv_file = './empathy_robotic_data/baseline/Llava_scenario_gt_l2.csv'
    output_file = "./results/Llava_scenario_gt_l2.json"
    
    evaluator.evaluate(csv_file=csv_file, output_file=output_file)