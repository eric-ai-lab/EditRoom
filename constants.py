import os

EDITROOM_DATA_FOLDER = os.getenv("EDITROOM_DATA_FOLDER", "./datasets")
print(
    f"EDITROOM_DATA_FOLDER is set to {EDITROOM_DATA_FOLDER}. "
    "If you want to change the path, set the EDITROOM_DATA_FOLDER environment variable."
)
PATH_TO_SCENE = os.path.join(EDITROOM_DATA_FOLDER, "3D-FRONT/3D-FRONT")
PATH_TO_MODEL = os.path.join(EDITROOM_DATA_FOLDER, "3D-FRONT/3D-FUTURE-model")
EDIT_DATA_FOLDER = os.path.join(EDITROOM_DATA_FOLDER, "editroom_dataset")
PATH_TO_PREPROCESS = os.path.join(EDITROOM_DATA_FOLDER, "preprocess")