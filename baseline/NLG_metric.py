import json
import csv
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu
from pycocoevalcap.cider.cider import Cider
import bert_score
from bert_score import score as bert_score_func
from transformers import AutoTokenizer

# Ensure you have downloaded the necessary NLTK data files
# nltk.data.path.append("./models/punkt/tokenizers/punkt")
# nltk.download('punkt', download_dir = "./models/")

class BLEUScore:
    def __init__(self,n):
        self.n = n
    def cal_similarity(self, response, gt, n=4):
        response_tokens = nltk.word_tokenize(response)
        gt_tokens = nltk.word_tokenize(gt)
        if n==1:
            score = sentence_bleu([gt_tokens], response_tokens, weights=(1.0, 0, 0, 0))
        else:
            score = sentence_bleu([gt_tokens], response_tokens)
        return score

    def score(self, response_dict, test_file_path, test_level=""):
        self.response_dict = response_dict
        with open(test_file_path, 'r', encoding='utf-8') as infile:
            self.test_data = json.load(infile)
            
        total_similarity = 0.0
        num = len(self.response_dict)
        
        for idx, test_data_item in tqdm(enumerate(self.test_data), total=len(self.test_data)):
            if test_level == "scenario":
                gt = test_data_item["scenario"]
            elif test_level == "high_level_plan":
                if test_data_item["rank"][0]==1:
                    gt = test_data_item["high_level_plan"]["0"]
                else:
                    gt = test_data_item["high_level_plan"]["1"]
            else:
                print("Wrong test level!")
                return
                
            if self.response_dict.get(f'{idx}'):
                response = self.response_dict[f'{idx}']
                print(response, "\n", gt)
                similarity = self.cal_similarity(response, gt, self.n)
                print(f"Similarity for index {idx}: {similarity}")
                total_similarity += similarity
                
        average_similarity = total_similarity / num if num > 0 else 0.0
        print(f"Average BLEU Score: {average_similarity}")


class CIDErScore:
    def __init__(self):
        self.cider_scorer = Cider()

    def cal_similarity(self, response, gt):
        response_tokens = ' '.join(nltk.word_tokenize(response))
        gt_tokens = ' '.join(nltk.word_tokenize(gt))
        print(gt_tokens,"\n",response_tokens)
        score, _ = self.cider_scorer.compute_score({0: [gt_tokens]}, {0: [response_tokens]})
        return score

    def score(self, response_dict, test_file_path, test_level=""):
        self.response_dict = response_dict
        with open(test_file_path, 'r', encoding='utf-8') as infile:
            self.test_data = json.load(infile)
            
        total_similarity = 0.0
        num = len(self.response_dict)
        
        for idx, test_data_item in tqdm(enumerate(self.test_data), total=len(self.test_data)):
            if test_level == "scenario":
                gt = test_data_item["scenario"]
            elif test_level == "high_level_plan":
                if test_data_item["rank"][0]==1:
                    gt = test_data_item["high_level_plan"]["0"]
                else:
                    gt = test_data_item["high_level_plan"]["1"]
            else:
                print("Wrong test level!")
                return
                
            if self.response_dict.get(f'{idx}'):
                response = self.response_dict[f'{idx}']
                similarity = self.cal_similarity(response, gt)
                print(f"Similarity for index {idx}: {similarity}")
                total_similarity += similarity
                
        average_similarity = total_similarity / num if num > 0 else 0.0
        print(f"Average CIDEr Score: {average_similarity}")

        
class BERTScore:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        
    def cal_similarity(self, response, gt, model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        response_tokens = tokenizer(response, max_length=512, truncation=True, padding=False, return_tensors="pt")
        gt_tokens = tokenizer(gt, max_length=512, truncation=True, padding=False, return_tensors="pt")
        
        truncated_response = tokenizer.decode(response_tokens['input_ids'][0], skip_special_tokens=True)
        truncated_gt = tokenizer.decode(gt_tokens['input_ids'][0], skip_special_tokens=True)
        print(response, "\n", truncated_response)
    
        P, R, F1 = bert_score_func([truncated_response], [truncated_gt], lang="en", verbose=False, model_type=self.model_dir, num_layers=12)
        return F1.mean().item() 

    def score(self, response_dict, test_file_path, test_level=""):
        self.response_dict = response_dict
        with open(test_file_path, 'r', encoding='latin') as infile:
            self.test_data = json.load(infile)
            
        total_similarity = 0.0
        num = len(self.response_dict)
        
        for idx, test_data_item in tqdm(enumerate(self.test_data), total=len(self.test_data)):
            if test_level == "scenario":
                gt = test_data_item["scenario"]
            elif test_level == "high_level_plan":
                if test_data_item["rank"][0]==1:
                    gt = test_data_item["high_level_plan"]["0"]
                else:
                    gt = test_data_item["high_level_plan"]["1"]
            else:
                print("Wrong test level!")
                return
                
            if self.response_dict.get(f'{idx}'):
                response = self.response_dict[f'{idx}']
                print(response, "\n", gt)
                similarity = self.cal_similarity(response, gt, model_dir=self.model_dir)
                print(f"Similarity for index {idx}: {similarity}")
                total_similarity += similarity
                
        average_similarity = total_similarity / num if num > 0 else 0.0
        print(f"Average BERTScore: {average_similarity}")
        

if __name__ == "__main__":
    csv_file = "Llava_scenario_gt_l2.csv"  
    response_dict = {}
    with open(csv_file, 'r', newline='', encoding='latin-1') as file:
        reader = csv.DictReader(file)
        for row in reader:
            response = row['response']
            idx = row['data_idx']
            if response is not None and response.strip():
                response_dict[f'{idx}'] = response
    test_file = "./dataset_scale_up/testset_100.json"
    # bleu_score = BLEUScore(1)
    # bleu_score.score(response_dict, test_file, test_level="scenario")    
    bert_score = BERTScore(model_dir = "./models/bert-base-uncased")
    bert_score.score(response_dict, test_file, test_level="high_level_plan")
    # cider_score = CIDErScore()
    # cider_score.score(response_dict, test_file, test_level="scenario")
