import os
from huggingface_hub import hf_hub_url
from constants import EDITROOM_DATA_FOLDER

url = hf_hub_url(repo_id="KZ-ucsc/EditRoom_dataset", filename="3D-FRONT.zip", repo_type="dataset")
os.system(f"wget {url} -P {EDITROOM_DATA_FOLDER} && cd {EDITROOM_DATA_FOLDER} && unzip 3D-FRONT.zip")

url = hf_hub_url(repo_id="KZ-ucsc/EditRoom_dataset", filename="preprocess.zip", repo_type="dataset")
os.system(f"wget {url} -P {EDITROOM_DATA_FOLDER} && cd {EDITROOM_DATA_FOLDER} && unzip preprocess.zip")

url = hf_hub_url(repo_id="KZ-ucsc/EditRoom_dataset", filename="objfeat_vqvae.zip", repo_type="dataset")
os.system(f"wget {url} -P {EDITROOM_DATA_FOLDER} && cd {EDITROOM_DATA_FOLDER} && unzip objfeat_vqvae.zip")