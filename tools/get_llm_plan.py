import os
import json
import argparse
import math
from tqdm import tqdm
import pickle
from copy import deepcopy
import ast
import requests
import transformers
import torch
import base64
import time
from src.data.threed_front import ThreedFront
from tools.utils import submit_batch_files, retrieve_batch_response

from constants import EDIT_DATA_FOLDER

api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "Please set the OPENAI_API_KEY environment variable"

def get_llm_plan(scene_data_folder, splits=["train", "test"], prompt_num_limit=50000):

    data_stats_path = os.path.join(scene_data_folder, "dataset_stats.txt")
    prompt_files = []
    for split in splits:
        edit_folder_path = os.path.join(scene_data_folder, f"{split}_dataset")
        if not os.path.exists(edit_folder_path):
            print(f"Edit data folder {edit_folder_path} does not exist.")
            continue
        
        command_file = os.path.join(scene_data_folder, f"batch_llm_command_{split}.json")
        if not os.path.exists(command_file):
            print(f"Command file {command_file} does not exist.")
            continue
        with open(command_file, "r") as f:
            command_data = json.load(f)

        scenes_dataset = ThreedFront.load_from_folder(edit_folder_path, path_to_train_stats=data_stats_path)
        class_labels = scenes_dataset.class_labels[:-2]
        class_labels = [c.replace("_", " ") for c in class_labels]
        batch_prompts = []
        uid_to_scene_index = {scene.uid: i for i, scene in enumerate(scenes_dataset)}

        for scene_id in tqdm(command_data, desc="Creating prompts"):
            if scene_id not in uid_to_scene_index:
                print(f"Scene {scene_id} not found in dataset. Please check the command data.")
                continue
            target_scene_index = uid_to_scene_index[scene_id]
            target_scene = scenes_dataset[target_scene_index]
            assert hasattr(target_scene, "command"), f"Scene {scene_id} does not have command, which should not happen."
            source_scene_id = target_scene.original_id
            source_scene = scenes_dataset[uid_to_scene_index[source_scene_id]]
        
            natural_command = command_data[scene_id]
            if type(natural_command) == list:
                natural_command = natural_command[0]

            message = construct_plan_prompt(source_scene, natural_command, class_labels)
            batch_prompts.append(construct_batch_element(message, scene_id))
        
        if len(batch_prompts) > prompt_num_limit:
            part_number = math.ceil(len(batch_prompts) / prompt_num_limit)
            part_size = len(batch_prompts) // part_number
            for i in range(part_number):
                start_index = i * part_size
                end_index = (i + 1) * part_size if i != part_number - 1 else len(batch_prompts)
                batch_prompts_part = batch_prompts[start_index:end_index]
                batch_prompts_file = os.path.join(scene_data_folder, f"batch_llm_plan_prompt_{split}_{i}.jsonl")
                with open(batch_prompts_file, "w") as f:
                    for batch_element in batch_prompts_part:
                        f.write(json.dumps(batch_element) + "\n")
                print(f"Saved prompts to {batch_prompts_file}")
                prompt_files.append(batch_prompts_file)
        else:
            batch_prompts_file = os.path.join(scene_data_folder, f"batch_llm_plan_prompt_{split}.jsonl")
            with open(batch_prompts_file, "w") as f:
                for batch_element in batch_prompts:
                    f.write(json.dumps(batch_element) + "\n")
            print(f"Saved prompts to {batch_prompts_file}")
            prompt_files.append(batch_prompts_file)
    return prompt_files

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def construct_plan_prompt(source_scene, instruction, class_labels, use_image=False):

    scene_description = source_scene.get_room_description()
    
    system_prompt = "Imagine you are a indoor room designer and you are using provided API to control the 3D models in the scene.\n" + \
        "Given one scene configuration and a command to edit the scene, you should use the provided APIs to do planning and achieve the target.\n" + \
        "All sizes and centroids in scene configurations are in meters. The angles are defined in degrees. The dimension sequence is [x,y,z]. Vertical angles are the angles along the y-axis.\n"+ \
        "Sizes are the half lengths of the bounding box along the x, y, and z axes when the vertical angle is zero.\n" + \
        "We define +x/-x as the right/left direction, +y/-y as the up/down direction, and +z/-z as the front/back direction.\n" + \
        "Positive angles are counterclockwise, and negative angles are clockwise.\n\n" + \
        "APIs:\n" + \
        "1. Rotate an object: ['Rotate', Target Object Description, Angle :(degrees)]\n" + \
        "2. Translate an object: ['Translate', Target Object Description, Direction :(x/z), Distance :(meters)]\n" + \
        "3. Scale an object: ['Scale', Target Object Description, Scale Factor]\n" + \
        "4. Replace an object: ['Replace', Source Object Description, Target Object Description]\n" + \
        "5. Add an object: ['Add', Target Object Description, (Relative Description, Relative Object Description)]\n" + \
        "6. Remove an object: ['Remove', Target Object Description]\n\n" + \
        "Matters needing attention:\n" + \
        "1. If there are multiple same objects in the scene and the command is related to the object, you should refer to the object locations.\n" + \
        "When you refer to the object locations, you should this format: (Relative Description, Relative Object Description). All reference should be append in the end of API lists.\n" + \
        "When you use add or remove command, you should refer to the object locations.\n" + \
        "Relative Description: [left, right, in front of, behind, above, below, closely left, closely right, closely in front of, closely behind]. 'closely' means the distance between two object centroids are less than 1 meters in x-z plane.\n" + \
        "For example, if you want to add a chair in front of the table, you should use the format: ['Add', 'chair', ('in front of', 'table')].\n" + \
        "At most add one relative description and one relative object description. Select the cloest one if there are more than two relative descriptions.\n" + \
        "The relative object description should be the same as the object description in the scene configuration.\n" + \
        "2. Translate, rotate, and scale commands should be executed in the order of scale, rotate, and translate.\n" + \
        "Translate should only work in the x/z direction. The distance should be one float number.\n" + \
        "3. When you scale an object, the object should be scaled uniformly. Scale factor should one float number.\n" + \
        "4. Replace object will only replace the object with the same class. Replace command will only change the object appearance, not the object poses and sizes.\n" + \
        "5. If Translate/Rotate/Scale commands can achieve the target, you should not use Replace/Add/Remove commands.\n" + \
        "6. If image is provided, you should use the image to help you understand the scene.\n" + \
        "7. Attempt to use the minimum number of commands to achieve the target.\n" + \
        f"8. If you want to add or replace object, you can only consider from these object classes: {json.dumps(class_labels)}.\n" + \
        "9. If you want to remove and add the object within the same class, you should use the replace command.\n" + \
        "10. Object descriptions should be detailed descriptions instead of class names. You can imagine the object descriptions if the object is not in the scene.\n" + \
        "11. Do not repeat the same API with the same objects.\n"+\
        "12. All apis should be able to converted to a list of strings and numbers, which can be directly processed by json.loads()\n\n"+\
        "For example:\n" + \
        "1. If you want to rotate a chair 90 degrees and there is only one chair in the scene, you should use the format: ['Rotate', 'chair', 90].\n" + \
        "2. If you want to add a chair in front of the wooden table, you should use the format: ['Add', 'chair', ('in front of', 'a wooden table')].\n" + \
        "3. If you want to remove a chair, you should use the format: ['Remove', 'chair'].\n" + \
        "4. If you want to replace a metal chair with a wooden one and this chair on the left of the bed with wooden design, you should use the format: ['Replace', 'the chair is metal', 'the chair is wooden', ('left', 'the bed is wooden')].\n\n" + \
        "Think about it step by step. Summarize the used apis at the end by lines. The final output format should be ***[api 1, api 2, ...]***.\n"
    
    prompt = "[Scene configurations]:\n" + scene_description + "\n" + \
        "[Command]:" + json.dumps(instruction) + "\n\n" + \
        "If there are multiple relative descriptions for one API, you should select the closest one.\n" + \
        "Checkout at the end to make sure output the final plan in the format of ***[api 1, api 2, ...]***.\n"
    
    if use_image:
        source_scene.get_blender_render("/tmp/blender_render", camera_dist=1.2, num_images=2, verbose=False)
        content = []
        for i in range(2):
            image_path = os.path.join("/tmp/blender_render", f"{i:03d}.png")
            if os.path.exists(image_path):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                    }
                })
        content.append({
            "type": "text",
            "text": prompt
        })
    else:
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
    message = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
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

def process_llm_plan(response_data_file, prompt_data_file):

    all_data = {}
    with open(response_data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            custom_id = data['custom_id']
            response = data["response"]['body']['choices'][0]['message']['content']
            all_data[custom_id] = response
    
    all_prompts = {}
    with open(prompt_data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            custom_id = data['custom_id']
            prompt = data['body']['messages']
            all_prompts[custom_id] = prompt

    save_file_name = "batch_llm_plan_train.json" if "train" in response_data_file else "batch_llm_plan_test.json"
    save_folder = os.path.dirname(response_data_file)
    save_path = os.path.join(save_folder, save_file_name)

    processed_all_data = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            processed_all_data = json.load(f)
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"attn_implementation":"flash_attention_2", "torch_dtype": torch.bfloat16}, device_map="auto")
    messages = [
        {"role": "system", "content": "Directly extract the command from the input. Ingore the analyze inside the input. Output format should be a list of strings and numbers. Remove the punctuation, like *, ouside the brackets"},
        {"role": "user", "content": """Input: To achieve the command of shrinking \"the modern cabinet with a light wood finish and white interior by half\", we need to scale the specified cabinet object by a factor of 0.5. Here's the step-by-step action plan:\n\n1. Identify the target object, which is the cabinet.\n2. Apply a scale transformation with a factor of 0.5.\n\nSince there is only one cabinet in the scene configuration, we do not need to refer to its location with relative descriptions.\n\nTherefore, the required API command is:\n1. Scale the cabinet by a factor of 0.5.\n\nThe output in the required format is:\n\n***[['Scale', 'the cabinet is a modern, minimalist design with a light wood finish and a white interior.', 0.5]]***"""},
        {"role": "assistant", "content": """Output: [['Scale', 'the cabinet is a modern, minimalist design with a light wood finish and a white interior.', 0.5]]"""},
        {"role": "user", "content": """Input: Let's start by identifying the target object in the scene and understanding the command given.\n\nThe target object based on the provided description is:\n- Object 11: a modern ceiling fan with a wooden base and a white light bulb\n\nThe command is to reduce the size of this object by 0.8 times of its current size.\n\nHere's a step-by-step plan to achieve the target with the minimum number of commands, considering the object already exists and only needs scaling:\n\n1. **Scaling the Target Object**: Reduce the size of the target object by 0.8 times.\n  \nHere's the final API command to achieve the target:\n\n```json\n[['Scale', 'the pendant lamp is a modern ceiling fan with a wooden base and a white light bulb', 0.8]]\n```\n\nThis is the minimal and most direct way to achieve the desired change in the scene configuration."""},
        {"role": "assistant", "content": """Output: [['Scale', 'the pendant lamp is a modern ceiling fan with a wooden base and a white light bulb', 0.8]]"""}
    ]
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for uid, command in tqdm(all_data.items()):
        assert uid in all_prompts, f"Prompt data not found for {uid}. Please check the prompt data file."
        if uid in processed_all_data:
            continue
        valid, react, proccessed_command = check_response_valid(command, pipeline, messages, terminators)
        if valid:
            processed_all_data[uid] = react
        else:
            react = reflect_invalid_command(command, react, proccessed_command, all_prompts[uid], pipeline, messages, terminators)
            if react is None:
                print(f"Failed to get a valid plan for {uid}. We will skip this scene.")
                continue
            processed_all_data[uid] = react
    
    with open(save_path, 'w') as f:
        json.dump(processed_all_data, f, indent=4)

def reflect_invalid_command(command, fail_reason, proccessed_command, prompts, pipeline, messages, terminators, max_retries=3):
    # Retry calling openai if the response is invalid
    for i in range(max_retries):
        assistant_context = {
                    "type": "text",
                    "text": command
                }
        user_message = {
                    "type": "text",
                    "text": f"Your response is invalid. The reason is: {fail_reason}. Please try again." if proccessed_command is None else f"Your response is invalid. The reason is: {fail_reason}. Please try again. The previous response after processing: {proccessed_command}"
                }
        prompts += [
            {
                "role": "assistant",
                "content": assistant_context
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        messages = {
            "model": "gpt-4o",
            "messages": prompts,
            "max_tokens": 2048
        }

        new_command = call_openai_api(messages)
        if new_command is None:
            print("Failed to get a response from OpenAI API. Please check your API key and network connection.")
            return None
        valid, react, proccessed_command = check_response_valid(new_command, pipeline, messages, terminators)
        if valid:
            return react
        else:
            command = new_command
            fail_reason = react
    
    return None


def check_response_valid(command, pipeline, messages, terminators):
    messages_i = deepcopy(messages)
    messages_i.append({"role": "user", "content": f"""Input: {command.lower()}"""})
    response = pipeline(messages_i, max_length=4096, eos_token_id=terminators, do_sample=True, temperature=0.1,)
    response = response[0]['generated_text'][-1]['content']
    # 1. Check if the response can be convert to string
    valid_response = response.replace("Output:", "").strip().lower().replace(";", "").replace("*", "").replace("].", "]").replace(")", "]").replace("(","[")
    try:
        valid_response = ast.literal_eval(valid_response)
    except Exception as e:
        return False, str(e), None
    
    # 2. Check if the response is a list
    if not isinstance(valid_response, list):
        return False, "The response is not a list", valid_response
    elif isinstance(valid_response[0], str):
        first_element = valid_response[0]
        if first_element in ["add", "remove", "replace", "scale", "rotate", "translate"]:
            valid_response = [valid_response]
        else:
            return False, "The first element of the response is not a valid action", valid_response
    
    # 3. Check if the response follows the format
    instructions = []
    for item in valid_response:
        try:
            instruction = convert_single_plan(item)
            instructions.append(instruction)
        except Exception as e:
            return False, str(e), valid_response
    return True, instructions, valid_response

def convert_single_plan(plan):
    def add_relative(relative):
        direction = relative[0]
        target = relative[1]
        if "right" in direction:
            relative = "right of"
        elif "left" in direction:
            relative = "left of"
        elif "front" in direction:
            relative = "in front of"
        else:
            relative = direction
        
        if "closely" in direction and not "closely" in relative:
            relative = "closely " + relative
        
        return f"location: ***{relative}*** {target}"
            
    if plan[0] == "add":
        assert len(plan) == 3, "The add plan should have 3 elements"
        target = plan[1]
        relative = plan[2]
        assert len(relative) == 2, "The relative should have 2 elements"
        relative_des = add_relative(relative)
        instruction = f"add object: {target}; {relative_des}."
    elif plan[0] == "remove":
        assert len(plan) in [2, 3], "The remove plan should have 2 or 3 elements"
        target = plan[1]
        if len(plan) == 3:
            relative = plan[2]
            assert len(relative) == 2, "The relative should have 2 elements"
            relative_des = add_relative(relative)
            instruction = f"remove object: {target}; {relative_des}."
        else:
            instruction = f"remove object: {target}."
    elif plan[0] == "translate":
        assert len(plan) in [4, 5], "The translate plan should have 4 or 5 elements"
        target = plan[1]
        direction = plan[2]
        distance = plan[3]
        assert direction in ['x', 'z'], "The direction should be x or z"
        assert type(distance) in [int, float], "The distance should be a number"
        direction_dict = {
            "x": "left" if distance < 0 else "right",
            "z": "front" if distance < 0 else "back",
        }
        distance = abs(distance)
        instruction = f"move object towards the ***{direction_dict[direction]}*** direction for {distance:.2f} meters: {target}"
        if distance > 1:
            instruction = "obviously " + instruction
        elif distance < 0.5:
            instruction = "slightly " + instruction

        if len(plan) == 5:
            relative = plan[4]
            assert len(relative) == 2, "The relative should have 2 elements"
            relative_des = add_relative(relative)
            instruction += f"; {relative_des}."
    elif plan[0] == "rotate":
        assert len(plan) in [3, 4], "The rotate plan should have 3 or 4 elements"
        target = plan[1]
        angle = plan[2]
        assert type(angle) in [int, float], "The angle should be a number"
        if abs(angle) >= 135:
            instruction = f"obviously rotate object {angle:.0f} degrees: {target}"
        elif abs(angle) <= 45:
            instruction = f"slightly rotate object {angle:.0f} degrees: {target}"
        else:
            instruction = f"rotate object {angle:.0f} degrees: {target}"
        if len(plan) == 4:
            relative = plan[3]
            assert len(relative) == 2, "The relative should have 2 elements"
            relative_des = add_relative(relative)
            instruction += f"; {relative_des}."
    elif plan[0] == 'scale':
        assert len(plan) in [3, 4], "The scale plan should have 3 or 4 elements"
        target = plan[1]
        scale = plan[2]
        assert type(scale) in [int, float], "The scale should be a number"
        if scale > 1:
            instruction = f"enlarge object by {scale:.1f} X: {target}"
            if scale > 1.3:
                instruction = "obviously " + instruction
        elif scale < 1:
            instruction = f"shrink object by {scale:.1f} X: {target}"
            if scale < 0.7:
                instruction = "obviously " + instruction
        else:
            instruction = None
        if len(plan) == 4:
            relative = plan[3]
            assert len(relative) == 2, "The relative should have 2 elements"
            relative_des = add_relative(relative)
            instruction += f"; {relative_des}."
    elif plan[0] == 'replace':
        assert len(plan) in [3, 4], "The replace plan should have 3 or 4 elements"
        source = plan[1]
        target = plan[2]
        instruction = f"replace source with target : [Source] {source}; [Target] {target}"
        if len(plan) == 4:
            relative = plan[3]
            assert len(relative) == 2, "The relative should have 2 elements"
            relative_des = add_relative(relative)
            instruction += f"; {relative_des}."
    else:
        raise ValueError(f"Invalid plan action: {plan}")
    assert instruction is not None, "Cannot process the instruction. Please check the plan."
    if instruction[-1] == "." and instruction[-2] == ".":
        instruction = instruction[:-1]
    return instruction

def call_openai_api(message, retries=3):
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    
    for i in range(retries):
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=message)
        response = response.json()
        if 'choices' in response.keys():
            desc = response['choices'][0]['message']['content']
            return desc
        else:
            time.sleep(2**i)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Train a generative model on bounding boxes"
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

    parser.add_argument(
        "--post_processing",
        action="store_true",
        help="Post-process the output"
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
            
            prompt_files = get_llm_plan(scene_data_folder, splits=args.splits)
            if len(prompt_files) == 0:
                print(f"No data found for {room_type}.")
                continue
            print("Submitting batch files to OpenAI...")
            path_to_id = submit_batch_files(prompt_files, api_key)
            path_to_id_save_path = os.path.join(scene_data_folder, f"batch_llm_plan_prompt_{room_type}_id.json")
            with open(path_to_id_save_path, 'w') as f:
                json.dump(path_to_id, f, indent=4)
            print(f"Saved path to id mapping to {path_to_id_save_path}")
    else:
        print("Post-processing the output...")
        for room_type in args.room_type:
            scene_data_folder = os.path.join(EDIT_DATA_FOLDER, f"threed_front_{room_type}")
            if not os.path.exists(scene_data_folder):
                raise FileNotFoundError(f"Edit data folder {scene_data_folder} does not exist. Please run edit_data_generator.py first.")
            batch_llm_output = os.path.join(scene_data_folder, f"batch_llm_plan_prompt_{room_type}_id.json")
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
                
                process_llm_plan(response_save_path, file_name)
                print(f"Processed {file_name} and saved to {response_save_path}")
        