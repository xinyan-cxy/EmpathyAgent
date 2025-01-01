import re, csv, json

legal_actions = {
    "get_toiletpaper_puton_bedroomtable",
    "get_glass_of_water_from_bathroom_puton_bedroomtable",
    "get_mug_of_water_puton_bedroomtable",
    "get_apple_puton_bedroomtable",
    "get_chicken_puton_bedroomtable",
    "get_radio_puton_bedroomtable",
    "get_box_puton_bedroomtable",
    "get_paper_puton_bedroomtable",
    "get_folder_puton_bedroomtable",
    "get_pillow_puton_bedroomtable",
    "get_wallphone_puton_bedroomtable",
    "get_cellphone_puton_bedroomtable",
    "get_kitchen_candle_puton_bedroomtable",
    "get_coffee_puton_bedroomtable",
    "get_breadslice_puton_bedroomtable",
    "get_book_puton_bedroomtable",
    "get_toiletpaper_puton_kitchentable",
    "get_glass_of_water_from_bathroom_puton_kitchentable",
    "get_mug_of_water_puton_kitchentable",
    "get_apple_puton_kitchentable",
    "get_chicken_puton_kitchentable",
    "get_radio_puton_kitchentable",
    "get_box_puton_kitchentable",
    "get_wallphone_puton_kitchentable",
    "get_cellphone_puton_kitchentable",
    "get_kitchen_candle_puton_kitchentable",
    "get_coffee_puton_kitchentable",
    "get_breadslice_puton_kitchentable",
    "switchon_bathroom_faucet",
    "switchon_radio",
    "switchoff_bedroom_tablelamp",
    "switchoff_bathroom_lights",
    "switchon_kitchen_candle",
    "switchon_stove",
    "switchon_computer",
    "switchon_tv",
    "open_fridge",
    "close_fridge",
    "sit_bed",
    "sit_bedroom_chair",
    "sit_bedroom_sofa",
    "sit_kitchen_bench",
    "cook_chicken_puton_bedroomtable",
    "cook_hot_water_puton_bedroomtable",
    "play_computer",
    "put_paper_into_folder_puton_bedroomtable",
    "put_book_into_bookshelf",
    "put_book_into_box_puton_bedroomtable",
    "put_apple_into_fridge_puton_bedroomtable",
    "put_mug_of_water_into_fridge_puton_bedroomtable",
    "None",
    "none",
    "dialogue"
}

def check_actions(input_string):
    actions = re.findall(r'<(.*?)>', input_string)
    
    correct_count = sum(1 for action in actions if action in legal_actions)
    total_count = len(actions)
    
    accuracy = (correct_count / total_count) * 10 if total_count > 0 else 0
    return accuracy

csv_file = './empathy_robotic_data/baseline/l3/Llava.csv'
output_file = "./results_fixed/Llava_l3.json"

with open(output_file, 'r') as f:
    results = json.load(f)

# Load data
response_dict = {}
with open(csv_file, 'r', newline='', encoding='latin-1') as file:
    reader = csv.DictReader(file)
    for row in reader:
        response = row['response']
        idx = row['data_idx']
        if response is not None and response.strip():
            response_dict[f'{idx}'] = response

for idx in range(100):
    if response_dict.get(f'{idx}') is None:
        continue
    input_string = response_dict[f'{idx}']
    accuracy = check_actions(input_string)
    print(accuracy)
    if accuracy != 10:
        print(input_string)
    # Save results after processing
    results[f"{idx}"]["legality"] = {
            "score": accuracy
            }

with open("./results_fixed/Llava_l3_new.json", 'w') as f:
    json.dump(results, f, indent=4)