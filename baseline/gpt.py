import base64
import os
import httpx
from openai import OpenAI

class GPT:
    def __init__(self, model_name):    
        self.client = OpenAI(
            base_url="", 
            api_key="",
            http_client=httpx.Client(
                base_url="",
                follow_redirects=True,
            ),
        )
        if model_name in ["gpt-4o", "gpt-4-turbo-2024-04-09", "gpt-4-turbo", "gpt-4-vision-preview"]:
            self.model_name = model_name
        else: 
            print("Wrong model name!")
            
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def base64_encode(self, input_path):
        base64Frames = []
        for filename in os.listdir(input_path):
            if filename.endswith(".png"):  # Check if the file is a PNG image
                filepath = os.path.join(input_path, filename)
                base64Frames.append(self.encode_image(filepath))  # Append base64 encoded image to the list
                if len(base64Frames) > 99:  # Break the loop if 500 frames are reached
                    break
        print(len(base64Frames), "frames read.")
        return base64Frames
            
    def generate(self, script_path, text_prompt):
        base64Frames = self.base64_encode(script_path)
        if self.model_name == "gpt-4o":
            content = [
                {"type": "text", "text": text_prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{x}", "resize": 768}}
                    for x in base64Frames[::10]
                ]
            ]
        else:
            content = [
                {"type": "text", "text": text_prompt},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{x}"}}
                    for x in base64Frames[::10]
                ]
            ]
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=300
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    
class GPT_text:
    def __init__(self, model_name):    
        self.client = OpenAI(
            base_url="", 
            api_key="",
            http_client=httpx.Client(
                base_url="",
                follow_redirects=True,
            ),
        )
        if model_name in ["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview", "gpt-3.5-turbo-0125"]:
            self.model_name = model_name
        else: 
            print("Wrong model name!")
            
    def generate(self, text_prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {"role": "user", "content": text_prompt}
            ],
            max_tokens=300
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content
    

if __name__ == "__main__":
    from inference_text import UNIFY_PROMPT, INPUT_PROMPT
    gpt = GPT_text(model_name = "gpt-3.5-turbo-0125")
    text_prompt = UNIFY_PROMPT + INPUT_PROMPT.format(character_info="Personality: Ambitious and unsure\n Profession: Aspiring Medical Office Manager\n Hobbies: Reading and cycling\n Social Relationships: In college, seeking career advice\n Life Experiences: Currently exploring different career paths in college, excited about the future but not fully decided.\n", 
    dialogue ="What am I really passionate about? I need to figure this out soon.", action="\"[Walktowards] <chair> (1)\", \"[Sit] <chair> (1)\"")
    print(gpt.generate(text_prompt))
    
