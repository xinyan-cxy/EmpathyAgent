from typing import Optional
import random 
import torch 
import torch.nn as nn 
import transformers
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig 
from transformers.dynamic_module_utils import get_class_from_dynamic_module 
import logging
import json 

class RandomRewardModel:
    def __init__(self):
        pass 
    def score(self, input):
        return random.random(0, 1)

def _get_reward_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = self.value_head(last_hidden_states).squeeze(-1)

            # left padding in training mode
            if self.training:
                reward = values[:, -1]
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                # normalize reward in eval mode
                if self.normalize_reward:
                    reward = (reward - self.mean) / self.std
            if return_output:
                return reward, outputs
            else:
                return reward

    return LLMForSequenceRegression


class LlaMaRewardModel(nn.Module):
    def __init__(self, 
                model_name_or_path,
                model_type = "reward", 
                source_data_path = "",
                source_character_path = "", 
                source_action_path = "",  
                bf_16 = True,
                lora_rank = 0,
                lora_alpha = 16,
                target_modules = None, 
                lora_dropout = 0,
                normalize_reward = False,
                use_flash_attention_2 = False,
                ds_config = None,
                init_value_head = False,
                device_map = None, 
                **kwargs):
        super().__init__()
        self.model, self.tokenizer = self.init_model(
            model_name_or_path,
            model_type,
            bf_16,
            lora_rank,
            lora_alpha,
            target_modules,
            lora_dropout,
            normalize_reward,
            use_flash_attention_2,
            ds_config,
            init_value_head,
            device_map,
            **kwargs
        )
        if torch.cuda.device_count() > 1: 
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.source_data_path = source_data_path 
        self.source_character_path = source_character_path

    def eval_empathy(self, human_sentence, model_generated_sentence):
        self.model.eval() 
        with torch.no_grad():
            human_ids, human_mask = self.prepare_inputs(human_sentence)
            model_generated_ids, model_generated_mask = self.prepare_inputs(model_generated_sentence)
            human_ids = human_ids.to(torch.cuda.current_device())
            human_mask = human_mask.to(torch.cuda.current_device())
            model_generated_ids = model_generated_ids.to(torch.cuda.current_device())
            model_generated_mask = model_generated_mask.to(torch.cuda.current_device())

            human_reward, model_generated_reward = self.concatenated_forward(
                human_ids, human_mask, model_generated_ids, model_generated_mask
            )
            return human_reward.item(), model_generated_reward.item() 

    def prepare_inputs(self, sentence):
            """        
            Args:
                sentence (str): 输入句子。
            Returns:
                torch.Tensor: 输入的ID张量。
                torch.Tensor: 输入的掩码张量。
            """
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            return input_ids, attention_mask

    def concatenated_forward(self, human_ids, human_mask, model_generated_ids, model_generated_mask):
            """        
            Args:
                human_ids (torch.Tensor): 人类标注句子的ID张量。
                human_mask (torch.Tensor): 人类标注句子的掩码张量。
                model_generated_ids (torch.Tensor): 模型生成句子的ID张量。
                model_generated_mask (torch.Tensor): 模型生成句子的掩码张量。
            """
            
            human_ids = human_ids.long()
            human_mask = human_mask.long()
            model_generated_ids = model_generated_ids.long()
            model_generated_mask = model_generated_mask.long()

            human_outputs = self.model(input_ids=human_ids, attention_mask=human_mask)
            model_generated_outputs = self.model(input_ids=model_generated_ids, attention_mask=model_generated_mask)

            #print("Human outputs:", human_outputs)
            #print("Model generated outputs:", model_generated_outputs)

            return human_outputs, model_generated_outputs

    def init_model(self, 
                model_name_or_path,
                model_type = "reward", 
                source_data_path = "",
                source_character_path = "", 
                source_action_path = "",  
                bf_16 = True,
                lora_rank = 0,
                lora_alpha = 16,
                target_modules = None, 
                lora_dropout = 0,
                normalize_reward = False,
                use_flash_attention_2 = False,
                ds_config = None,
                init_value_head = False,
                device_map = None, 
                **kwargs):

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code = True) 
        config.normalize_reward = normalize_reward 
        config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
        auto_model_name, pretrainened_model_name = "", ""
        try: 
            base_class = AutoModel._model_mapping[type(config)]
            base_pretrained_class = base_class.__base__ 
            cls_class = _get_reward_model(base_pretrained_class, base_class) 
        except Exception as e: 
            print("Failed to load from AutoModel, construct from modelling file.")
            module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

            if "AutoModel" not in config.auto_map: 
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else: 
                auto_model_name = config.auto_map["AutoModel"].split(".")[1] 
            pretrainened_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"
            
            print(f"Base Model Class : {auto_model_name}, Pretrained Model Class : {pretrained_model_name}")
            base_pretrained_class = get_class_from_dynamic_module(
                f"{module_file}.{pretrained_model_name}", model_name_or_path
            )
            base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
            cls_class = _get_reward_model(base_pretrained_class, base_class)
        
        model = cls_class.from_pretrained(
            model_name_or_path, 
            config = config,
            trust_remote_code = True, 
            torch_dtype = torch.bfloat16 if bf_16 else "auto",
            device_map = device_map,
            **kwargs
        )
        #self.loss_fn = PairWiseLoss() if loss_fn_name == "sigmoid" else LogExpLoss()
        #self.margin_loss = True
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        logging.info(f"Load Reward Model : {cls_class.__name__} from {model_name_or_path} Success !")
        return model, tokenizer

   
    def score(self, input_str_list): 
        json_datas = json.load(open(self.source_data_path, "r", encoding='latin-1'))
        character_json_data = json.load(open(self.source_character_path, "r"))
        # assert len(json_data) == len(input_str_list), "Input string list should have the same length as the json data."
        json_data = json_datas[:len(input_str_list)]
        input_idx = 0 
        reward_diff_list, human_reward_list, model_generated_reward_list = [], [], []
        for data in json_data: 
            character_id = data["character_id"]
            character_info = character_json_data[str(character_id)]
            character_prompt = "The character info is as follows:" + character_info + "."
            scenario = data["scenario"]
            scenerio_prompt = "The scenario is as follows: " + scenario + "."
            dialogue = data["dialogue"]
            dialogue_prompt = "The human dialogue in the scenario is as follows: " + dialogue + "." 
            prefix_prompt = character_prompt + scenerio_prompt + dialogue_prompt
            robot_response = data["empathy_goal_nl"]["0"][0] if data["rank"][0] > data["rank"][1] else data["empathy_goal_nl"]["1"][0]
            human_annotate_sentence = prefix_prompt + "Robot Response is : " + robot_response
            model_annotate_sentence = prefix_prompt + "Robot Response is : " + input_str_list[input_idx]
            input_idx += 1
            human_reward, model_generated_reward = self.eval_empathy(human_annotate_sentence, model_annotate_sentence)
            reward_diff = model_generated_reward - human_reward
            print(f"Human Reward : {human_reward}, Model Generated Reward : {model_generated_reward}. Model Generated Reward - Human Reward : {reward_diff}")
            reward_diff_list.append(reward_diff)
            human_reward_list.append(human_reward)
            model_generated_reward_list.append(model_generated_reward)
        return reward_diff_list, human_reward_list, model_generated_reward_list

if __name__ == "__main__":
    model_name_or_path = "./OpenRLHF/examples/scripts/ckpt/7b_llama"
    llama_rm = LlaMaRewardModel(
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
    input_nl =  "switchon radio, get mug of water and put it on bedroomtable, say:\"Let me turn on some soothing music and get you a mug of water to help you relax and clear your mind. It's completely normal to feel overwhelmed with so many options, but you'll find your path in due time. Take it one step at a time.\""
    input_score = llama_rm.score([input_nl])