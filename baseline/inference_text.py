import random  
import json 
import os 
import csv
import time
from tqdm import tqdm
from rewardmodel import RandomRewardModel as rm
from gemini import Gemini
from gpt import GPT_text
import argparse

UNIFY_PROMPT = """
Assume you are an empathatic robot which can understand the emotion behind the human actions in different scenarios and make empathatic response to the human action. Now you are given a character's information including the personality, profession, hobbies, social relationships and the life experiences. You are also given the person's action list and the dialogue the person makes in the scenario.  Your job is as follows:
1. Understand what the person is trying to do in the action list.
2. Understand the person's current emotion state based on the action list and the dialogue.
3. Make VALID empathatic response based on the input action and the dialogue. 
4. Formulate your response with the format :  <action_1>, ..., <action_n>, <dialogue>:DIALOGUE_CONTENT. ALL the action in response MUST be selected from the following legal action space and the dialogue MUST be provided at LAST. You can refer to the example for more information.

The legal action space is listed as follows : 
1. fetch objects(description: fetch objects and put them on bedroom table.):
get_toiletpaper_puton_bedroomtable
get_glass_of_water_from_bathroom_puton_bedroomtable
get_mug_of_water_puton_bedroomtable
get_apple_puton_bedroomtable
get_chicken_puton_bedroomtable 
get_radio_puton_bedroomtable
get_box_puton_bedroomtable
get_paper_puton_bedroomtable
get_folder_puton_bedroomtable
get_pillow_puton_bedroomtable
get_wallphone_puton_bedroomtable
get_cellphone_puton_bedroomtable
get_kitchen_candle_puton_bedroomtable
get_coffee_puton_bedroomtable
get_breadslice_puton_bedroomtable
get_book_puton_bedroomtable
get_toiletpaper_puton_kitchentable
get_glass_of_water_from_bathroom_puton_kitchentable
get_mug_of_water_puton_kitchentable
get_apple_puton_kitchentable
get_chicken_puton_kitchentable 
get_radio_puton_kitchentable
get_box_puton_kitchentable
get_wallphone_puton_kitchentable
get_cellphone_puton_kitchentable
get_kitchen_candle_puton_kitchentable
get_coffee_puton_kitchentable
get_breadslice_puton_kitchentable

2. Utilizing furnitures (description: changeing the state of the furniture wthiout moving it):
switchon_bathroom_faucet
switchon_radio
switchoff_bedroom_tablelamp
switchoff_bathroom_lights
switchon_kitchen_candle
switchon_stove
switchon_computer
switchon_tv 
open_fridge (The fridge is empty now)
close_fridge

3. Sit(description: sit on something):
sit_bed
sit_bedroom_chair
sit_bedroom_sofa
sit_kitchen_bench

4.combination action(description: processing multi-step actions):
cook_chicken_puton_bedroomtable
cook_hot_water_puton_bedroomtable
play_computer
put_paper_into_folder_puton_bedroomtable
put_book_into_bookshelf
put_book_into_box_puton_bedroomtable
put_apple_into_fridge_puton_bedroomtable
put_mug_of_water_into_fridge_puton_bedroomtable

5.Do Nothing:
None
----------------------------------------------------------------
Now the action list is [action]. The chacter information is [character_info]. The dialogue made by the person in the scenario is [dialogue]. 
Correct Example Answer: 
1. <get_glass_of_water_from_bathroom_puton_bedroomtable>, <get_folder_puton_bedroomtable>, <switchon_radio>, <dialogue>:"I figured you may need a hydration break and a place to store your coin details. I also switched on the radio for some relaxing music."
2. <get_mug_of_water_puton_bedroomtable>, <switchon_tv>, <dialogue>:"You've had a long day. Why don't you take a moment to unwind? I've brought you some water and turned on the TV for a bit of relaxation."
----------------------------------------------------------------
Now the action list is [action]. The chacter information is [character_info]. The dialogue made by the person in the scenario is [dialogue]. 
Wrong Example Answer:
1. <dialogue>:"I see you need some fresh toilet paper, let me fetch you one." <get_toiletpaper_puton_bedroomtable> 
Explanation : <dialogue> can not be front of the <action> 
2. <get_book_puton_bedroomtable>, <dialogue>:"You must feel very tired now. Please read some books to relax."
Explanation : <get_book_puton_bedroomtable> is not a action in legal action space. 
--------------------------------------------------------------
NOTE:
1. All the actions in response MUST be chosen from the action space provided above.
2. The dialogue MUST be provided after the action. 
3. DO NOT provide the repeated action. 
4. If you do not want to do any action, you should answer <None>. But you still need to answer with the dialogue following None.
"""

INPUT_PROMPT = """ 

Now, the chacter information is {character_info}. The action made by the person is {action}. The dialogue made by the person in the scenario is "{dialogue}". Your response is : 
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4-vision-preview",
        help="Choose between 'gpt-4o','gpt-4-turbo', 'gpt-4-turbo-2024-04-09','gpt-4-vision-preview','gemini-1.0-pro-vision-001','gpt-3.5-turbo-0125'",
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

def inference(model, model_name = "", empathy_scenario_data_path = "", character_data_path = "", action_data_path = ""):
    response_list = []
    empathy_scenario_data = json.load(open(empathy_scenario_data_path, 'r', encoding='utf-8')) 
    character_data = json.load(open(character_data_path,'r'))
    action_data = json.load(open(action_data_path,'r'))
    csv_file_path = model_name + "_gt.csv"
    existing_indices = load_existing_indices(csv_file_path)
    
    with open(csv_file_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not existing_indices:  # Write the header only if the file is empty or does not exist
            writer.writerow(["data_idx", "response"])
        for idx, data in tqdm(enumerate(empathy_scenario_data)):
            if idx in existing_indices:
                continue
            character_id, dialogue, action_id = data["character_id"], data["dialogue"], data["action_id"]
            action = action_data[str(action_id)]
            character_info = character_data[str(character_id)]
            input_prompt = UNIFY_PROMPT + INPUT_PROMPT.format(character_info = character_info, dialogue = dialogue, action = action)
            # print(INPUT_PROMPT.format(character_info = character_info, dialogue = dialogue, action = action))
            try:
                response = model.generate(input_prompt)
                response_list.append(response)
                writer.writerow([idx, response])  
            except Exception as e:
                if 'Quota exceeded' in str(e):
                    print("Man, what can I say? I'm out of quota. Exiting now.")
                    raise e  
                else:
                    raise e   
            """
            #For local mllm: 
            response = model.generate(video_input, input_prompt).text
            response_list.append(response)
            writer.writerow([idx, response]) 
            """
    print(f"Inference Done for {model_name}!")


def visualize(scores):
    pass 

if __name__ == "__main__":
    args = parse_args()
    
    if args.model_name == "gemini-1.0-pro-vision-001":
        #project_id = "" 
        location = "us-central1" 
        gemini = Gemini(project_id = project_id, location = location, model_name = "gemini-1.0-pro-vision-001")
        inference(gemini, 
                "Gemini", 
                empathy_scenario_data_path = "./dataset_scale_up/scenario_10k.json", 
                character_data_path = "./dataset_scale_up/character.json", 
                video_path = "./dataset_scale_up/video")
        
    elif args.model_name in ["gpt-4o", "gpt-4-turbo-2024-04-09", "gpt-4-vision-preview", "gpt-3.5-turbo-0125","gpt-4-turbo"]:
        gpt = GPT_text(model_name = args.model_name)
        inference(gpt,
                args.model_name,
                empathy_scenario_data_path = "./dataset_scale_up/scenario_10k.json", 
                character_data_path = "./dataset_scale_up/character.json", 
                action_data_path = "./dataset_scale_up/action_list.json")
        
    else:
        print("Model name is wrong!")
        

