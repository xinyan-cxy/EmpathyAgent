import json

with open("./results/gpt-4o_scenario_gt_l2.json", "r") as f:
    data = json.load(f)
    
level = 2

if level == 3:
    scores = {
        "adaptability": [],
        "association": [],
        "coherence": [],
        "emotion_reg": [],
        "emotional_com_l23": [],
        "helpfulness": [],
        "individual": [],
        "legality": [],
    }
elif level == 2:
        scores = {
        "adaptability": [],
        "association": [],
        "coherence": [],
        "emotion_reg": [],
        "emotional_com_l23": [],
        "helpfulness": [],
        "individual": []
    }
else:
    scores = {
        "association": [],
        "coherence_l1": [],
        "emotional_com_l1": [],
        "individual": [],
    }

for index in data:
    for key in scores.keys():
        if key in data[index]:
            score = data[index][key]["score"]
            if score is not None:  
                scores[key].append(score)

averages = {key: sum(value) / len(value) for key, value in scores.items() if value}
overall_average = sum(averages.values()) / len(averages) if averages else 0

print("averages:", {key: f"{avg:.2f}" for key, avg in averages.items()})
print(f"overall_average: {overall_average:.2f}")