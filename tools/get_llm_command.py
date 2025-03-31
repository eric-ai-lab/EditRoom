import os
import json
import argparse
import math
from tqdm import tqdm
import pickle
from copy import deepcopy
import ast
import re
import transformers
import torch
from src.data.threed_front import ThreedFront
from tools.utils import submit_batch_files, retrieve_batch_response

from constants import EDIT_DATA_FOLDER

api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "Please set the OPENAI_API_KEY environment variable"

def get_llm_command(scene_data_folder, splits=["train", "test"], prompt_num_limit=50000):

    objects_dataset_path = os.path.join(scene_data_folder, f"{config['filter_fn']}_objects.pkl")
    with open(objects_dataset_path, "rb") as f:
        objects_dataset = pickle.load(f)

    data_stats_path = os.path.join(scene_data_folder, "dataset_stats.txt")
    prompt_files = []
    for split in splits:
        edit_folder_path = os.path.join(scene_data_folder, f"{split}_dataset")
        if not os.path.exists(edit_folder_path):
            print(f"Edit data folder {edit_folder_path} does not exist.")
            continue
        scenes_dataset = ThreedFront.load_from_folder(edit_folder_path, path_to_train_stats=data_stats_path)
        batch_prompts = []
        uid_to_scene_index = {scene.uid: i for i, scene in enumerate(scenes_dataset)}
        for scene in tqdm(scenes_dataset):
            if not hasattr(scene, "command"):
                continue
            source_scene_index = uid_to_scene_index[scene.original_id]
            source_scene = scenes_dataset[source_scene_index]
            message = construct_prompt(source_scene, scene, objects_dataset)
            batch_element = construct_batch_element(message, scene.uid)
            batch_prompts.append(batch_element)
        
        if len(batch_prompts) > prompt_num_limit:
            part_number = math.ceil(len(batch_prompts) / prompt_num_limit)
            part_size = len(batch_prompts) // part_number
            for i in range(part_number):
                start_index = i * part_size
                end_index = (i + 1) * part_size if i != part_number - 1 else len(batch_prompts)
                batch_prompts_part = batch_prompts[start_index:end_index]
                batch_prompts_file = os.path.join(scene_data_folder, f"batch_llm_command_prompt_{split}_{i}.jsonl")
                with open(batch_prompts_file, "w") as f:
                    for batch_element in batch_prompts_part:
                        f.write(json.dumps(batch_element) + "\n")
                print(f"Saved prompts to {batch_prompts_file}")
                prompt_files.append(batch_prompts_file)
        else:
            batch_prompts_file = os.path.join(scene_data_folder, f"batch_llm_command_prompt_{split}.jsonl")
            with open(batch_prompts_file, "w") as f:
                for batch_element in batch_prompts:
                    f.write(json.dumps(batch_element) + "\n")
            print(f"Saved prompts to {batch_prompts_file}")
            prompt_files.append(batch_prompts_file)
    return prompt_files

def construct_prompt(source_scene, target_scene, object_dataset):

    scene_description = source_scene.get_room_description()
    instruction = target_scene.command
    
    all_descriptions = []
    for instruc in instruction.split("[JID]"):
        if "[/JID]" in instruc:
            obj_jid = instruc.split("[/JID]")[0]
            obj = object_dataset.get_furniture_by_jid(obj_jid)
            description = obj.description()
            if description.endswith("."):
                description = description[:-1]
            all_descriptions.append(instruc.replace(f"{obj_jid}[/JID]", description))
        else:
            all_descriptions.append(instruc)
    
    instruction = "".join(all_descriptions)
    instruction += "."
    instruction = instruction.split('\n')

    target_object_index = target_scene.uid.split("_")[-2].split("-")[-1]
    
    system_prompt = "Given scene configurations and templated commands, you should write new commands using natural language and spatial referring.\n" + \
        "Templated commands will be in the 'action: target_object' format. If the location is provided in the templated commands, it can be considered as a hint for the target object's location compared to the existing object in the scene.\n" + \
        "All sizes and centroids in scene configurations are in meters. The angles are defined in degrees. The dimension sequence is [x,y,z]. Vertical angles are the angles along the y-axis.\n"+ \
        "Sizes are the half lengths of the bounding box along the x, y, and z axes when the vertical angle is zero.\n" + \
        "We define +x/-x as the right/left direction, +y/-y as the up/down direction, and +z/-z as the front/back direction.\n" + \
        "When you design new commands, please refer to the spatial relations between objects in the scene.\n" + \
        "When you design new commands, please consider correctness, conciseness, and naturalness.\n" + \
        "You should attempt to make your command need reasoning.\n" + \
        "If there are duplicate target objects in the scene, you should refer to object locations by relative spatial relations with one unique object in the scene.\n" + \
        "If there are multiple templated commands, you should consider them as the same command with different representations.\n" + \
        "If templated commands indicate to add an object where there is already a similar object, you should indicate this is about adding new object in your command.\n" + \
        "Enlarge and shrink in the command should be uniform.\n" + \
        "You can add object descriptions according to the scene configurations, commands, and the image (if provided).\n" + \
        "For example:\n" + \
        "Example1:\n" + \
        "[Templated commands]:[\"move object towards the ***left*** direction for 1 meters: a white bed with a red and white plaid comforter and a red and white plaid pillow.\"]\n" + \
        "If there is a table (only one table inside the scene) on the left side of the bed and length of bed is 2 meters, you can write: [\"move the white bed with red and white plaid towards the table around 1 meters.\"] or [\"move the bed towards left direction by half of bed length.\"]\n" + \
        "Example2:\n" + \
        "[Templated commands]:[\"move object towards the ***left*** direction for 0.5 meters: a wooden nightstand.\"]\n" + \
        "If there is a bed parallel to the nightstand and moving to the left will make nightstand more closer to the bed headboard, you can write: [\"move the nightstand closer to the bed headboard by 0.5 meters\"].\n" + \
        "Example3:\n" + \
        "[Templated commands]:[\"replace source with target : [Source] a white bed; [Target] a brown bed.\"]\n" + \
        "You can write: [\"replace the white bed with a brown bed.\"]\n" + \
        "Example4:\n" + \
        "[Templated commands]:[\"add object: a white bed; location: ***right*** a wardrobe.\"]\n" + \
        "If there is a wardrobe in the scene, you can write: [\"add a white bed on the right side of the wardrobe.\"]\n" + \
        "Now you can start to design new commands based on the scene configurations and templated commands. You can supplement object descriptions on the command.\n" + \
        "Think about it step by step and summarize your commands in the end. The final output format should be '###[natural command 1, natural command 2, ...]###', which is a list of strings and can be processed by ast.literal_eval() or json.loads().\n"
    
    prompt = "[Scene configurations]:\n" + scene_description + "\n" + \
        "[Templated commands]:" + json.dumps(instruction) + "\n" + \
        f"Hint: The target object is the Object_{target_object_index}.\n\n" +\
        "Think about it step by step and summarize your commands in the end. The final output format should be '###[natural command 1, natural command 2, ...]###', which is a list of strings.\n"
    
    message = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": prompt # prompt
                        },
                    ]
            }
        ],
        "max_tokens": 2048
    }
    return message

def construct_batch_element(message, uid):
    batch_element = {
        "custom_id": uid,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": message
    }
    return batch_element

def process_natural_command(data_file):

    def extract_and_convert_to_list(s):
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, s)
        if match:
            bracketed_content = match.group(1).strip()
            # Replace curly quotes with straight quotes
            bracketed_content = bracketed_content.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
            # Special handling for single content inside brackets
            try:
                # Attempt to evaluate it as a list
                result_list = ast.literal_eval(f'[{bracketed_content}]')
                if isinstance(result_list, list):
                    if len(result_list)==0 or not isinstance(result_list[0], str) or len(result_list[0])<10:
                        return None
                    return result_list
            except (ValueError, SyntaxError):
                # If evaluation fails, return the content as a single-item list
                return None
        return None

    all_data = {}
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            custom_id = data['custom_id']
            response = data["response"]['body']['choices'][0]['message']['content']
            all_data[custom_id] = response

    processed_all_data = {}
    to_llm_process = []
    for k, v in all_data.items():
        response = v.replace('\n', '').replace('(','[').replace(')',']').replace('{', '[').replace('}',']').lower().split("###")
        valid_response = None
        for i in range(len(response), 1, -1):
            valid_response = response[i-1]
            if len(valid_response)==0:
                continue
            valid_response = extract_and_convert_to_list(valid_response)
            if valid_response is not None:
                break
        if valid_response is None:
            to_llm_process.append((k, v))
        else:
            processed_all_data[k] = valid_response
    
    messages = [
            {"role": "system", "content": "Directly extract the natural command from the input. Ingore the analyze inside the input. Remove the exact location number inside the command, like [2.47, -3.71]. Remove object pronoun inside the command, like (object 0) Output format should be a list of strings."},
            {"role": "user", "content": """Input: Given the scene configurations and the templated commands, we see that the target object is Object 2: the nightstand, described as "a modern, sleek, and minimalist gray rectangular box with a flat top and bottom."\n\nNext, let\'s consider how we can reference the nightstand based on its relations with other objects in the scene.\n\n- Object 0 (double_bed) is located at [1.77, 0.0, 3.19], which is towards the front relative to the coordinate origin.\n- Object 1 (wardrobe) is at [1.99, 0.0, 1.59], on the front and similar right relative position.\n- Object 2 (nightstand) is at [2.65, 0.0, 4.41], in front and towards the right of the bed (Object 0).\n- Object 3 (pendant_lamp) is above all the objects at [1.5, 2.03, 2.96].\n\nWe can create a natural language command for rotating the nightstand (Object 2) by referring to its relative position to the bed (Object 0) since there are no duplicate target objects.\n\nThe templated commands hint at rotating the nightstand by ±30 degrees.\n\nHere\'s a natural command based on the scene.\n\n###["Rotate the modern, sleek, and minimalist gray nightstand 30 degrees towards the right, keeping it in its position next to the bed."]###\n\nThis command leverages the relational position to the bed (Object 0) to provide a natural and concise instruction."""},
            {"role": "assistant", "content": """Output: ["Rotate the modern, sleek, and minimalist gray nightstand 30 degrees towards the right, keeping it in its position next to the bed."]"""},
            {"role": "user", "content": """Input: Let\'s analyze the scene configuration and templated command step by step:\n\n1. **Target Object Identification**:\n   - The target object is **Object 5**, which is a "black geometric chandelier with three lights."\n\n2. **Relative Movement**:\n   - The command is to move the chandelier 0.05 meters (5 cm) to the right.\n\n3. **Object\'s Current Position**:\n   - The current position (centroid) of the pendant lamp (Object 5) is at coordinates: `[-0.36, 1.46, 0.44]`.\n\n4. **Ensuring No Collision**:\n   - The horizontal movement is along the `x` axis, so updating `x` coordinate from `-0.36` to `-0.31`.\n   - There are no objects mentioned near the pendant lamp directly along the `x` axis, thus moving it should not result in any collision.\n\nCombining these points, we can give a clear and concise natural command for moving the object.\n\n### \n1. \'Move the black geometric chandelier slightly to the right by 5 centimeters.\'\n2. \'Slide the black chandelier 5 centimeters towards the right."""},
            {"role": "assistant", "content": """Output: ["Move the black geometric chandelier slightly to the right by 5 centimeters.", "Slide the black chandelier 5 centimeters towards the right."]"""}
        ]
    batch_messages = []
    k_list = []
    for k, v in to_llm_process:
        messages_i = deepcopy(messages)
        messages_i.append({"role": "user", "content": f"""Input: {v.lower()}"""})
        batch_messages.append(messages_i)
        k_list.append(k)

    if batch_messages:
        batch_size = 16
        model_id = "meta-llama/Llama-3.1-8B-Instruct"

        pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"attn_implementation":"flash_attention_2", "torch_dtype": torch.bfloat16}, device_map="auto")
        # terminators = [
        #     pipeline.tokenizer.eos_token_id,
        #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]
        pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[-1]
        pipeline.tokenizer.padding_side = "left"
        pattern = r'\[.*\]'
        for i in tqdm(range(0, len(batch_messages), batch_size)):
            batch_messages_i = batch_messages[i:i+batch_size]
            k_list_i = k_list[i:i+batch_size]
            responses = pipeline(batch_messages_i, max_new_tokens=256, do_sample=True, temperature=0.1, batch_size=batch_size)
            for k, response in zip(k_list_i, responses):
                response_content = response[0]['generated_text'][-1]['content']
                match = re.search(pattern, response_content)
                if match:
                    bracket_content = match.group(0)
                    try:
                        extracted_list = json.loads(bracket_content)
                        processed_all_data[k] = extracted_list
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {response_content}")
                else:
                    print("No content found within brackets.")

    save_file_name = "batch_llm_command_train.json" if "train" in data_file else "batch_llm_command_test.json"
    save_folder = os.path.dirname(data_file)
    save_path = os.path.join(save_folder, save_file_name)
    with open(save_path, 'w') as f:
        json.dump(processed_all_data, f, indent=4)

    print('Number of data:', len(processed_all_data))

def convert_to_python_list(data_string):
    # Remove any leading or trailing whitespace
    data_string = data_string.strip().replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    
    # If the string is already a valid Python literal, use ast.literal_eval to convert it
    try:
        python_list = ast.literal_eval(data_string)
    except (ValueError, SyntaxError):
        # If literal_eval fails, handle unquoted list elements
        # Remove brackets and split the string by commas
        elements = re.findall(r'\b\w+\b', data_string)
        python_list = elements
    
    return python_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Train a generative model on bounding boxes"
        )
    
    parser.add_argument(
        "--post_processing",
        action="store_true",
        help="Post-process the output"
    )

    parser.add_argument(
        "--room_type",
        default=["bedroom", "livingroom", "diningroom"],
        nargs="+",
        help="The room types to be generated"
    )

    parser.add_argument(
        "--splits",
        default=["train", "test"],
        nargs="+",
        help="The splits to be generated"
    )

    args = parser.parse_args()

    if not args.post_processing:
        print("Constructing prompts for LLM...")
        for room_type in args.room_type:
            config = {
                "filter_fn":                 f"threed_front_{room_type}",
                "min_n_boxes":               -1,
                "max_n_boxes":               -1,
                "path_to_invalid_scene_ids": "configs/invalid_threed_front_rooms.txt",
                "path_to_invalid_bbox_jids": "configs/black_list.txt",
                "annotation_file":           f"configs/{room_type}_threed_front_splits.csv"
            }
            scene_data_folder = os.path.join(EDIT_DATA_FOLDER, config['filter_fn'])
            if not os.path.exists(scene_data_folder):
                raise FileNotFoundError(f"Edit data folder {scene_data_folder} does not exist. Please run edit_data_generator.py first.")
            
            prompt_files = get_llm_command(scene_data_folder, splits=args.splits)
            if len(prompt_files) == 0:
                print(f"No data found for {room_type}.")
                continue
            print("Submitting batch files to OpenAI...")
            path_to_id = submit_batch_files(prompt_files, api_key)
            path_to_id_save_path = os.path.join(scene_data_folder, f"batch_llm_command_prompt_{room_type}_id.json")
            with open(path_to_id_save_path, 'w') as f:
                json.dump(path_to_id, f, indent=4)
            print(f"Saved path to id mapping to {path_to_id_save_path}")
    else:
        print("Post-processing the output...")
        for room_type in args.room_type:
            scene_data_folder = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}")
            if not os.path.exists(scene_data_folder):
                raise FileNotFoundError(f"Edit data folder {scene_data_folder} does not exist. Please run edit_data_generator.py first.")
            batch_llm_output = os.path.join(scene_data_folder, f"batch_llm_command_prompt_{room_type}_id.json")
            with open(batch_llm_output, 'r') as f:
                path_to_id = json.load(f)
            responses = retrieve_batch_response(path_to_id, api_key)
            for file_name, response in responses.items():
                if response is None:
                    print(f"Error: {file_name} does not have response.")
                    continue
                response_save_path = os.path.join(scene_data_folder, f"{file_name.split('.')[0]}_response.jsonl")
                with open(response_save_path, "wb") as f:
                    f.write(response.content)
                
                process_natural_command(response_save_path)
                print(f"Processed {file_name} and saved to {response_save_path}")
        