# EditRoom Dataset Generation

We need to first create scene editing pairs with template commands and then convert them to natural language by ChatGPT.

## 1. Creating Editing pairs
The created dataset will be saved under `EDIT_DATA_FOLDER` in [constants.py](../constants.py). Please change it if you want another folder.

```bash
python3 --room_type bedroom --num_max_pre_room 20
```

The room type also can be `diningroom` or `livingroom`. If you change to those rooms, please set `num_max_pre_room` to a larger number, like `40`, since those rooms have less configurations but have more furnitures per room.

## 2. Creating LLM Commands

First, we will generate batch prompts for OpenAI Batch API to generate natural langauge commands. Since we only need LLM commands for evaluation, we only use `test` split here. Please change to `--splits train test` if you also want to generate natural language commands for training set.

```bash
python3 tools/get_llm_command.py --room_type bedroom --splits test
```

After waiting for LLM generation, we will get back the responses and postprocess them.

```bash
python3 tools/get_llm_command.py --room_type bedroom --splits test --post_processing
```

Then, you should see `batch_llm_command_test.json` under `threed_front_bedroom` folder.

## 3. Creating LLM Plans

Similarily, we will create batch prompts for OpenAI Batch API to generate editing plans.

```bash
python3 tools/get_llm_plan.py --room_type bedroom --splits test
```

After waiting for LLM generation, we will get back the responses and postprocess them.

```bash
python3 tools/get_llm_plan.py --room_type bedroom --splits test --post_processing
```

Then, you should see `batch_llm_plan_test.json` under `threed_front_bedroom` folder.

