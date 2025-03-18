import json
from typing import Dict, List, Any, Optional
import csv

class EmpathyScorer:
    def __init__(self, result_path: str = "./results/gpt-4o_scenario_gt_l2.json", level: int = 2):
        self.result_path = result_path
        self.level = level
        self.data = None
        self.scores = {}
        self.averages = {}
        self.overall_average = 0.0
    
    def initialize_score_dict(self) -> None:
        if self.level == 3:
            self.scores = {
                "adaptability": [],
                "association": [],
                "coherence": [],
                "emotion_reg": [],
                "emotional_com_l23": [],
                "helpfulness": [],
                "individual": [],
                "legality": [],
            }
        elif self.level == 2:
            self.scores = {
                "adaptability": [],
                "association": [],
                "coherence": [],
                "emotion_reg": [],
                "emotional_com_l23": [],
                "helpfulness": [],
                "individual": []
            }
        else:  # level 1
            self.scores = {
                "association": [],
                "coherence_l1": [],
                "emotional_com_l1": [],
                "individual": [],
            }
    
    def collect_scores(self) -> None:
        for index in self.data:
            for key in self.scores.keys():
                if key in self.data[index]:
                    score = self.data[index][key]["score"]
                    if score is not None:
                        self.scores[key].append(score)
    
    def calculate_averages(self) -> None:
        self.averages = {key: sum(value) / len(value) for key, value in self.scores.items() if value}
        self.overall_average = sum(self.averages.values()) / len(self.averages) if self.averages else 0
        
    def save_results(self, filename: str) -> None:
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Score"])
            for key, avg in self.averages.items():
                writer.writerow([key, round(avg, 2)])
                print(f"{key}: {round(avg, 2)}")
            writer.writerow(["overall_average", round(self.overall_average, 2)])
            print(f"overall_average: {round(self.overall_average, 2)}")
    
    def get_results(self) -> Dict[str, Any]:
        return {
            "averages": {key: round(avg, 2) for key, avg in self.averages.items()},
            "overall_average": round(self.overall_average, 2)
        }
        
    def run(self) -> Dict[str, Any]:
        with open(self.result_path, "r") as f:
            self.data = json.load(f)
        self.initialize_score_dict()
        self.collect_scores()
        self.calculate_averages()
        return self.get_results()


if __name__ == "__main__":
    result_path = "./results/gpt-4o_scenario_gt_l2.json"
    level = 2
    
    scorer = EmpathyScorer(result_path, level)
    results = scorer.run()
    scorer.print_results()