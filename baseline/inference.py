import random  
import json 
import os 
import csv
import time
from tqdm import tqdm
from pathlib import Path  
import sys   
sys.path.insert(0, str(Path(__file__).parent.parent.resolve() / "reward_model_metric"))
from rewardmodel import LlaMaRewardModel as llamarm
from gemini import Gemini
from gpt import GPT
import argparse

UNIFY_PROMPT = """
Assume you are an empathatic robot which can understand the emotion behind the human actions in different scenarios and make empathatic response to the human action. Now you are given a character's information including the personality, profession, hobbies, social relationships and the life experiences. You are also given a video recording the person's behaviours and the dialogue the person makes in the scenario.  Your job is as follows:
1. Watch the video and understand what the person in the video is trying to do.
2. Understand the person's current emotion state based on the video content and the dialogue the person makes in the scenario.
3. Make VALID empathatic response based on the video content and the dialogue you have read. 
4. Formulate your response with the format :  <action_1>, ..., <action_n>, <dialogue>:DIALOGUE_CONTENT. ALL the action MUST be selected from the following legal action space and the dialogue MUST be provided at LAST. You can refer to the example for more information.

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
Now the video Input is [VIDEO]. The chacter information is [character_info]. The dialogue made by the person in the scenario is [dialogue]. 
Correct Example Answer: 
1. <get_glass_of_water_from_bathroom_puton_bedroomtable>, <get_folder_puton_bedroomtable>, <switchon_radio>, <dialogue>:"I figured you may need a hydration break and a place to store your coin details. I also switched on the radio for some relaxing music."
2. <get_mug_of_water_puton_bedroomtable>, <switchon_tv>, <dialogue>:"You've had a long day. Why don't you take a moment to unwind? I've brought you some water and turned on the TV for a bit of relaxation."
----------------------------------------------------------------
Now the video Input is [VIDEO]. The chacter information is [character_info]. The dialogue made by the person in the scenario is [dialogue]. 
Wrong Example Answer:
1. <dialogue>:"I see you need some fresh toilet paper, let me fetch you one." <get_toiletpaper_puton_bedroomtable> 
Explanation : <dialogue> can not be front of the <action> 
2. <get_book_puton_bedroomtable>, <dialogue>:"You must feel very tired now. Please read some books to relax."
Explanation : <get_book_puton_bedroomtable> is not a action in legal action space. 
--------------------------------------------------------------
NOTE:
1. All the actions MUST be chosen from the action space provided above.
2. The dialogue MUST be provided after the action. 
3. DO NOT provide the repeated action. 
4. If you do not want to do any action, you should answer <None>. But you still need to answer with the dialogue following None.
"""

INPUT_PROMPT = """ 
Now the video Input is attached. The chacter information is {character_info}. The dialogue made by the person in the scenario is {dialogue}. Your response is : 
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-1.0-pro-vision-001",
        help="Choose between 'gpt-4o','gpt-4-turbo-2024-04-09','gpt-4-vision-preview','gemini-1.0-pro-vision-001'",
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

def inference(model, model_name = "", empathy_scenario_data_path = "", character_data_path = "", action_data_path = "", video_path = "", script_path = ""):
    response_list = []
    empathy_scenario_data = json.load(open(empathy_scenario_data_path, 'r', encoding='utf-8')) 
    character_data = json.load(open(character_data_path,'r'))
    
    csv_file_path = model_name + ".csv"
    existing_indices = load_existing_indices(csv_file_path)
    
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
            input_prompt = UNIFY_PROMPT + INPUT_PROMPT.format(character_info = character_info, dialogue = dialogue)
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

def llama_reward_model_eval(inference_response_list):
    """
    Args:
        - input_response_list: (_type_): Inference Response List returned from inference function. 
    Note : The order for Inference Result must be alligned with the source_data_path in rank_10k_gpt4_new.json 
    """
    model_name_or_path = "./OpenRLHF/examples/scripts/ckpt/7b_llama"
    llama_rm = llamarm(
        model_name_or_path,
        source_data_path = "./dataset_scale_up/rank_10k_gpt4_new.json",
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

def llama_reward_average_score(human_reward_list):
    if not human_reward_list: 
        return 0
    total_score = sum(human_reward_list)  
    average_score = total_score / len(human_reward_list)  
    return average_score

def visualize(input_score):
    pass 

if __name__ == "__main__":
    args = parse_args()
    
    if args.model_name == "gemini-1.0-pro-vision-001":
        project_id = "1" 
        location = "us-central1" 
        gemini = Gemini(project_id = project_id, location = location, model_name = "gemini-1.0-pro-vision-001")
        inference(gemini, 
                "Gemini", 
                empathy_scenario_data_path = "./dataset_scale_up/testset_100.json", 
                character_data_path = "./dataset_scale_up/character.json", 
                video_path = "./dataset_scale_up/video")
    elif args.model_name in ["gpt-4o", "gpt-4-turbo-2024-04-09", "gpt-4-vision-preview"]:
        gpt = GPT(model_name = args.model_name)
        inference(gpt,
                  args.model_name,
                  empathy_scenario_data_path = "./dataset_scale_up/testset_100.json", 
                character_data_path = "./dataset_scale_up/character.json", 
                  script_path = "./dataset_scale_up/scripts")
    else:
        print("Model name is wrong!")



