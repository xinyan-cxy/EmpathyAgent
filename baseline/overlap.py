import re
import json
import csv
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Overlap:
    def __init__(self, mode = ""):    
        self.mode = mode 
        
    def find_subgoals(self, text):
        pattern = r'<(.*?)>' 
        matches = re.findall(pattern, text)
        filtered_matches = [match for match in matches if match.lower() != 'dialogue']
        return filtered_matches
    
    def find_subgoals_nl(self, text):
        parts = text.split(", say", 1)
        first_part = parts[0].strip()
        subgoals = [item.strip() for item in first_part.split(',')]
        # print(subgoals)
        return subgoals

    def cal_acc(self, response_subgoals, gt_subgoals):
        matched_count = sum(1 for subgoal in gt_subgoals if subgoal in response_subgoals)
        accuracy = 2*matched_count / (len(response_subgoals)+len(gt_subgoals)) 
        return accuracy

    def score(self, response_dict, test_file_path):
        self.response_dict = response_dict
        with open(test_file_path, 'r', encoding='latin-1') as infile:
            self.test_data = json.load(infile)
            
        total_accuracy = 0.0
        num = len(self.response_dict)
        
        if self.mode == "nl":
            for idx, test_data_item in tqdm(enumerate(self.test_data)):
                if test_data_item["rank"][0] == 1:
                    gt = test_data_item["empathy_goal_nl"]["0"]
                else: 
                    gt = test_data_item["empathy_goal_nl"]["1"]
                gt_subgoals = self.find_subgoals_nl(gt[0])
                if self.response_dict.get(f'{idx}'):
                    response = self.response_dict[f'{idx}']
                    response_subgoals = self.find_subgoals_nl(response)
                    accuracy = self.cal_acc(response_subgoals, gt_subgoals)
                    # print(accuracy)
                    total_accuracy += accuracy
        else:
            for idx, test_data_item in tqdm(enumerate(self.test_data), total=len(self.test_data)):
                if test_data_item["rank"][0] == 1:
                    gt = test_data_item["empathy_goal"]["0"]
                else: 
                    gt = test_data_item["empathy_goal"]["1"]
                gt_subgoals = self.find_subgoals(gt)
                if self.response_dict.get(f'{idx}'):
                    response = self.response_dict[f'{idx}']
                    response_subgoals = self.find_subgoals(response)
                    accuracy = self.cal_acc(response_subgoals, gt_subgoals)
                    # print(accuracy)
                    total_accuracy += accuracy
                
        average_accuracy = total_accuracy / num if num > 0 else 0.0
        print(f"Average Overlapping Accuracy: {average_accuracy}")
        return average_accuracy
        
class LCS:
    def __init__(self, mode = ""):    
        self.mode = mode

    def find_subgoals(self, text):
        pattern = r'<(.*?)>'
        matches = re.findall(pattern, text)
        filtered_matches = [match for match in matches if match.lower() != 'dialogue']
        return filtered_matches
    
    def find_subgoals_nl(self, text):
        parts = text.split(", say", 1)
        first_part = parts[0].strip()
        subgoals = []
        for item in first_part.split(','):
            new_item = "_".join(item.strip().split(' '))
            subgoals.append(new_item)
        # print(subgoals)
        return subgoals

    def lcs_length(self, X, Y):
        m = len(X)
        n = len(Y)
        L = [[0] * (n + 1) for i in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        
        return L[m][n]

    def cal_similarity(self, response, gt_subgoals):
        lcs_len = self.lcs_length(response, gt_subgoals)
        max_len = max(len(response), len(gt_subgoals))
        similarity = lcs_len / max_len if max_len > 0 else 0.0
        return similarity

    def score(self, response_dict, test_file_path):
        self.response_dict = response_dict
        with open(test_file_path, 'r', encoding='latin-1') as infile:
            self.test_data = json.load(infile)
            
        total_similarity = 0.0
        num = len(self.response_dict)
        
        if self.mode == "nl":
            for idx, test_data_item in tqdm(enumerate(self.test_data), total=len(self.test_data)):
                if test_data_item["rank"][0] == 1:
                    gt = test_data_item["empathy_goal_nl"]["0"]
                else:
                    gt = test_data_item["empathy_goal_nl"]["1"]
                gt_subgoals = self.find_subgoals_nl(gt[0])
                if self.response_dict.get(f'{idx}'):
                    response = self.response_dict[f'{idx}']
                    response_subgoals = self.find_subgoals_nl(response)
                    similarity = self.cal_similarity(response_subgoals, gt_subgoals)
                    # print(f"Similarity for index {idx}: {similarity}")
                    total_similarity += similarity
        else:
            for idx, test_data_item in tqdm(enumerate(self.test_data), total=len(self.test_data)):
                if test_data_item["rank"][0] == 1:
                    gt = test_data_item["empathy_goal"]["0"]
                else:
                    gt = test_data_item["empathy_goal"]["1"]
                gt_subgoals = self.find_subgoals(gt)
                if self.response_dict.get(f'{idx}'):
                    response = self.response_dict[f'{idx}']
                    response_subgoals = self.find_subgoals(response)
                    similarity = self.cal_similarity(response_subgoals, gt_subgoals)
                    # print(f"Similarity for index {idx}: {similarity}")
                    total_similarity += similarity
                
        average_similarity = total_similarity / num if num > 0 else 0.0
        print(f"Average LCS Similarity: {average_similarity}")
        return average_similarity
        
class TF_IDF:
    def __init__(self, mode = ""):    
        self.mode = mode
        self.vectorizer = TfidfVectorizer()

    def find_subgoals(self, text):
        pattern = r'<(.*?)>'
        matches = re.findall(pattern, text)
        filtered_matches = [match for match in matches if match.lower() != 'dialogue']
        return filtered_matches
    
    def find_subgoals_nl(self, text):
        parts = text.split(", say", 1)
        first_part = parts[0].strip()
        subgoals = []
        for item in first_part.split(','):
            new_item = "_".join(item.strip().split(' '))
            subgoals.append(new_item)
        # print(subgoals)
        return subgoals

    def cal_similarity(self, response, gt):
        documents = [response, gt]
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]

    def score(self, response_dict, test_file_path):
        self.response_dict = response_dict
        with open(test_file_path, 'r', encoding='latin-1') as infile:
            self.test_data = json.load(infile)
            
        total_similarity = 0.0
        num = len(self.response_dict)
        
        if self.mode == "nl":
            for idx, test_data_item in tqdm(enumerate(self.test_data), total=len(self.test_data)):
                if test_data_item["rank"][0] == 1:
                    gt = test_data_item["empathy_goal_nl"]["0"]
                else:
                    gt = test_data_item["empathy_goal_nl"]["1"]
                gt_subgoals = ' '.join(self.find_subgoals_nl(gt[0]))
                if response_dict.get(f'{idx}'):
                    response = self.response_dict[f'{idx}']
                    response_subgoals = ' '.join(self.find_subgoals_nl(response))
                    similarity = self.cal_similarity(response_subgoals, gt_subgoals)
                    # print(f"Similarity for index {idx}: {similarity}")
                    total_similarity += similarity
        else:
            for idx, test_data_item in tqdm(enumerate(self.test_data), total=len(self.test_data)):
                if test_data_item["rank"][0] == 1:
                    gt = test_data_item["empathy_goal"]["0"]
                else:
                    gt = test_data_item["empathy_goal"]["1"]
                gt_subgoals = ' '.join(self.find_subgoals(gt))
                if response_dict.get(f'{idx}'):
                    response = self.response_dict[f'{idx}']
                    response_subgoals = ' '.join(self.find_subgoals(response))
                    similarity = self.cal_similarity(response_subgoals, gt_subgoals)
                    # print(f"Similarity for index {idx}: {similarity}")
                    total_similarity += similarity
                
        average_similarity = total_similarity / num if num > 0 else 0.0
        print(f"Average TF_IDF Similarity: {average_similarity}")
        return average_similarity


if __name__ == "__main__":
    csv_file = "./l3/llama3_instruct2.csv"  
    response_dict = {}
    with open(csv_file, 'r', newline='', encoding='latin-1') as file:
        reader = csv.DictReader(file)
        for row in reader:
            response = row['response']
            idx = row['data_idx']
            if response is not None and response.strip():
                response_dict[f'{idx}'] = response
    test_file = "./dataset/testset_100.json"
    overlap = Overlap()
    overlap.score(response_dict, test_file)
    lcs = LCS()
    lcs.score(response_dict, test_file)
    tf_idf = TF_IDF()
    tf_idf.score(response_dict, test_file)
    
        
    
    
    
    
    
            


