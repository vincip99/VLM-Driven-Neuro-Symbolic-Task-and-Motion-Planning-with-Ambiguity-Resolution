"""
This file is used to connect to the VLM server and send prompt to it. It receives the response back and check its schema.
"""
# import json
# from pydantic import ValidationError
# from planner.schema import RobotTask

# from behavior_tree import vocab

import io
import copy
import ollama
from typing import List
from . import ASSETS_DIR
from PIL import Image

class VlmChat:
    def __init__(self, vlm_name="qwen2.5vl:3b"):
        self.vlm_name = vlm_name
        self.images = []
        self.messages = []
        self.reset_chat()

    def set_image(self, images: list, reset_chat: bool = True):
        """
        Set the list of images for the VLM input
        """
        self.images = images
        if reset_chat:
            self.reset_chat()
        return

    def add_message(self, role: str, content: str):
        """
        Add a message to the chat history, expressing the role of the message.
        """
        assert role in ["user", "assistant"]    # Check if the role is valid (user, assistant (is the ai chatbot))
        self.messages.append({"role": role, "content": content})
        return

    def reset_chat(self):
        """
        Reset the chat history
        """
        self.messages = []
        return

    def clean_vlm_code(self, text: str, lang: str) -> str:
        """
        Extract code from LLM output wrapped in ```<lang> ... ``` blocks.
        Falls back to clean_json or raw text if code tags are missing.
        """
        try:
            return self.clean_pddl(text)
        except ValueError:
            if lang == "json":
                return self.clean_json(text)
            return text.strip()

    def clean_json(self, text):
        """
        Helper to extract the first valid JSON object from local model outputs.
        """
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find('{')
        if start == -1:
            return text
            
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                
            if brace_count == 0:
                return text[start:i+1]
                
        end = text.rfind('}') + 1
        return text[start:end] if end > start else text

    def clean_pddl(self, text: str) -> str:
        """
        General domain-agnostic extractor for PDDL code blocks.
        Extracts PDDL from ```pddl / ```markdown blocks or locates (define ...) expressions directly.
        """
        for lang_tag in ["```pddl", "```markdown", "```"]:
            if lang_tag in text:
                parts = text.split(lang_tag)
                for part in parts[1:]:
                    if "```" in part:
                        block = part.split("```", 1)[0].strip()
                        if "(define" in block:
                            return block
                    elif "(define" in part:
                        return part.strip()

        start_idx = text.find("(define")
        if start_idx != -1:
            sub = text[start_idx:]
            balance = 0
            for i, char in enumerate(sub):
                if char == '(':
                    balance += 1
                elif char == ')':
                    balance -= 1
                    if balance == 0:
                        return sub[:i+1].strip()
            return sub.strip()

        return text.strip()

    def inference(self, messages, generate_kwargs: dict = None, do_sample: bool = True, format_schema: dict = None) -> str:
        """
        Run the inference with the given messages and max_retries
        """
        # Ollama api use HTTP so it requires images as raw bytes
        processed_images = []
        for img in self.images:
            if isinstance(img, Image.Image):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                processed_images.append(img_byte_arr.getvalue())
            else:
                processed_images.append(img)

        ollama_messages = copy.deepcopy(messages)

        # Add images to last message
        if processed_images:
            for msg in reversed(ollama_messages):
                if msg["role"] == "user":
                    msg["images"] = processed_images
                    break

        options = {"temperature": 0.8 if do_sample else 0.0}
        options.update(generate_kwargs or {})
        options.update({
                "num_ctx": 8192  # Increases the context window to 8K tokens
            })

        response = ollama.chat(
            model=self.vlm_name,
            messages=ollama_messages,
            options=options,
            format=format_schema
        )

        return response['message']['content'].strip()
            
    def detect(self, objects: List[str]) -> List[str]:
        """
        Detect objects in the image.
        """
        out_messages = []
        for obj in objects:
            in_message = {"role": "user", "content": f"Find the {obj}."}
            text_out = self.inference([in_message])
            out_messages.extend([in_message, {"role": "assistant", "content": text_out}])
        return out_messages


    def step_chat(self, user_text: str):
        """
        Add a message to the chat history and run the inference.
        """
        self.add_message("user", user_text)
        text_out = self.inference(self.messages)
        self.add_message("assistant", text_out)
        return text_out



if __name__ == "__main__":
    ASSETS_DIR.IMAGES
    image = Image.open(ASSETS_DIR.IMAGES / "real" / "test" / "5rhU25AdQW4jADxhp8EYuq.jpeg")
    image = image.reduce(4)

    vlm_chat = VlmChat()
    vlm_chat.set_image([image])
    for _ in range(2):
        user_text = input("Type the next message: ")
        text_out = vlm_chat.step_chat(user_text)
        print(text_out)
    out = vlm_chat.detect(["mug", "red_marker"])
    print(out)

    print("done")


    # def _build_system_prompt(self, scene_objects: list, valid_goals) -> str:
    #     # We pass the actual Pydantic schema to the VLM so it knows exactly what fields to output
        
    #     return f"""You are a robotic planner. You must output a JSON object with these exact fields:
    #             - "reasoning": A short string explaining your plan.
    #             - "skills": A list of objects, where each object has:
    #                 - "skill": must be "pick" or "place".
    #                 - "target_object": one of {', '.join(scene_objects)}.
    #                 - "goal": one of {', '.join(valid_goals)}.

    #             Example:
    #             {{"reasoning": "Pick red can then place in bin", "skills": [{{"skill": "pick", "target_object": "red_can", "goal": "None"}}, {{"skill": "place", "target_object": "red_can", "goal": "sorting_bin"}}]}}
                
    #             Do not output the JSON schema. Output ONLY the JSON object.
    #             """
    
    # def generate_plan(self, image_b64, command, scene_objects, valid_goals, max_retries=3):
    #     system_prompt = self._build_system_prompt(scene_objects, valid_goals)
    #     reflection_history = [] 

    #     for attempt in range(max_retries):
    #         # Prepare messages for Ollama
    #         messages = [
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": command, "images": [image_b64]}
    #         ]
            
    #         if reflection_history:
    #             messages.append({"role": "assistant", "content": reflection_history[-1]['failed_output']})
    #             messages.append({"role": "user", "content": f"Validation Error: {reflection_history[-1]['error']}. Fix the JSON."})

    #         # Call Ollama
    #         response = ollama.chat(model=self.vlm_name, messages=messages)
    #         raw_response = response['message']['content']

    #         print(f"\n--- RAW LLAVA OUTPUT ---")
    #         print(raw_response)
    #         print(f"------------------------\n")
            
    #         # Clean the output
    #         cleaned_json = self._clean_json_output(raw_response)

    #         try:
    #             plan = RobotTask.model_validate_json(cleaned_json)
    #             print(f"Success on attempt {attempt + 1}")
    #             return plan
                
    #         except (ValidationError, json.JSONDecodeError) as e:
    #             print(f"Attempt {attempt + 1} failed validation. Reflecting...")
    #             reflection_history.append({
    #                 'failed_output': raw_response,
    #                 'error': str(e)
    #             })

    #     return None

    # def generate_behavior_tree(self, image_b64, scene_dict, instruction, max_retries=3):
    #     system_prompt = vocab.build_system_prompt()
    #     user_prompt = vocab.build_user_prompt(scene_dict, instruction)
    #     reflection_history = [] 

    #     for attempt in range(max_retries):
    #         messages = [
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt, "images": [image_b64]}
    #         ]
            
    #         if reflection_history:
    #             messages.append({"role": "assistant", "content": reflection_history[-1]['failed_output']})
    #             messages.append({"role": "user", "content": f"Validation Error: {reflection_history[-1]['error']}. Fix the JSON."})

    #         # Call Ollama
    #         response = ollama.chat(model=self.vlm_name, messages=messages)
    #         raw_response = response['message']['content']

    #         print(f"\n--- RAW LLAVA OUTPUT ---")
    #         print(raw_response)
    #         print(f"------------------------\n")
            
    #         # Clean the output
    #         cleaned_json = self._clean_json_output(raw_response)

    #         try:
    #             plan = json.loads(cleaned_json)
    #             vocab.validate_plan(plan["main_sequence"])
    #             print(f"Success on attempt {attempt + 1}")
    #             return plan
                
    #         except Exception as e:
    #             print(f"Attempt {attempt + 1} failed validation. Reflecting...")
    #             reflection_history.append({
    #                 'failed_output': raw_response,
    #                 'error': str(e)
    #             })

    #     return None