import os
import cv2
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from numpy import ndarray
from shapely.geometry import Polygon, Point
import math
import re
from openai import OpenAI

def preprocess_edits(dataset, obj_dataset, num_max_pre_room = 10):
    # add modified scenes of object replace and pose change operations to the original scene list, record numbers    
    dataset.n_original_scenes = len(dataset.scenes)
    obj_dictbylabel = {}
    objmodeljid_set = set()
    for obj in obj_dataset.objects:
        obj_class = obj.label
        if obj_class not in obj_dictbylabel.keys():
            obj_dictbylabel[obj_class] = [obj]
            objmodeljid_set.add(obj.model_jid)
        elif obj.model_jid not in objmodeljid_set:
            obj_dictbylabel[obj_class].append(obj)
            objmodeljid_set.add(obj.model_jid)
    add_remove_scene_list= []
    replace_scene_list, posechange_scene_list = [], []
    render_scenes = False
    for i in tqdm(range(dataset.n_original_scenes)):
        if render_scenes:
            render_scene(f'{i:06d}', dataset.scenes[i])
        current_scene = dataset.scenes[i]
        relation_dicts = add_localization(current_scene)
        current_scene.relation_dicts = relation_dicts
        current_room_count = 0
        while current_room_count < num_max_pre_room:
            object_index = np.random.permutation(len(current_scene.bboxes))
            for j in object_index:
                removed_scene, add_scene = add_remove(current_scene, i, j, relation_dicts=relation_dicts)
                removed_scene.uid += f"_remove-{j}_{current_room_count}"
                add_scene.uid += f"_add-{j}_{current_room_count}"
                removed_scene.original_id = current_scene.uid
                add_scene.original_id = removed_scene.uid
                add_remove_scene_list.append(removed_scene)
                add_remove_scene_list.append(add_scene)
                current_room_count += 2

                ret, scene = object_replace(current_scene, i, j, obj_dictbylabel, render=render_scenes, relation_dicts=relation_dicts)
                scene.uid += f"_replace-{j}_{current_room_count}"
                scene.original_id = current_scene.uid
                if ret is True:
                    replace_scene_list.append(scene)
                    current_room_count += 1
                    opposite_scene = get_opposite_data(current_scene, scene)
                    replace_scene_list.append(opposite_scene)
                    current_room_count += 1
                for n in range(3):
                    ret, scene, change_number = pose_change(current_scene, i, j, mode=n, render=render_scenes, relation_dicts=relation_dicts)
                    scene.uid += f"_pose-{n}-{j}_{current_room_count}"
                    scene.original_id = current_scene.uid
                    if ret is True:
                        posechange_scene_list.append(scene)
                        current_room_count += 1
                        opposite_scene = get_opposite_data(current_scene, scene, change_number)
                        posechange_scene_list.append(opposite_scene)
                        current_room_count += 1

                if current_room_count >= num_max_pre_room:
                    break
        
    if len(dataset.scenes) != dataset.n_original_scenes:
        dataset.scenes = dataset.scenes[:dataset.n_original_scenes]
    dataset.n_add_remove_scenes = len(add_remove_scene_list)
    dataset.n_replace_scenes = len(replace_scene_list)
    dataset.n_posechange_scenes = len(posechange_scene_list)
    dataset.scenes += add_remove_scene_list
    dataset.scenes += replace_scene_list
    dataset.scenes += posechange_scene_list

    return dataset

def get_opposite_data(original_scene, edit_scene, change_number=None):
    edit_command = edit_scene.command
    opposite_scene = deepcopy(original_scene)
    opposite_scene.original_id = edit_scene.uid
    if "_pose" in edit_scene.uid:
        if "enlarge" in edit_command:
            new_change_number = 1 / change_number
            edit_command = edit_command.replace('enlarge', 'shrink').replace(f'{change_number:.1f}', f'{new_change_number:.1f}')
        elif "shrink" in edit_command:
            new_change_number = 1 / change_number
            edit_command = edit_command.replace('shrink', 'enlarge').replace(f'{change_number:.1f}', f'{new_change_number:.1f}')
        elif "move" in edit_command:
            if "***front*** direction" in edit_command:
                edit_command = edit_command.replace("***front*** direction", "***back*** direction")
            elif "***back*** direction" in edit_command:
                edit_command = edit_command.replace("***back*** direction", "***front*** direction")
            elif "***left*** direction" in edit_command:
                edit_command = edit_command.replace("***left*** direction", "***right*** direction")
            elif "***right*** direction" in edit_command:
                edit_command = edit_command.replace("***right*** direction", "***left*** direction")
        elif "rotate" in edit_command:
            edit_command = edit_command.split('\n')[0]
            edit_command = edit_command.replace(f'{change_number:.0f} degrees', f'{-change_number:.0f} degrees')
    elif "_replace" in edit_scene.uid:
        pattern = r'\[JID\](.*?)\[/JID\]'
        jids = re.findall(pattern, edit_command)
        edit_command = edit_command.replace(f"[Source] [JID]{jids[0]}[/JID]; [Target] [JID]{jids[1]}[/JID]", f"[Source] [JID]{jids[1]}[/JID]; [Target] [JID]{jids[0]}[/JID]")
    
    opposite_scene.uid = edit_scene.uid+'-opposite'
    opposite_scene.command = edit_command
    return opposite_scene

def add_localization(current_scene):
    all_jids = np.array([mesh.model_jid for mesh in current_scene.bboxes])
    unqiue_obj_dict = {}
    for obj in current_scene.bboxes:
        jid = obj.model_jid
        if (all_jids==obj.model_jid).sum() == 1:
            unqiue_obj_dict[jid] = [obj.corners(), obj.label]
    
    relation_dicts = {}
    for i, obj in enumerate(current_scene.bboxes):
        jid = obj.model_jid
        corners = obj.corners()
        label = obj.label
        relations = []
        for key, value in unqiue_obj_dict.items():
            if key == jid:
                continue
            target_corners, target_label = value
            relation_str, distance = compute_loc_rel(corners, target_corners, label, target_label)
            if relation_str is None:
                continue
            relations.append([key, relation_str, distance])
        sorted_relations = sorted(relations, key=lambda x: x[2])
        if sorted_relations:
            relation_dicts[i] = sorted_relations[0]
    return relation_dicts

def compute_loc_rel(corners1: ndarray, corners2: ndarray, name1: str, name2: str):
    assert corners1.shape == corners2.shape == (8, 3), "Shape of corners should be (8, 3)."

    center1 = corners1.mean(axis=0)
    center2 = corners2.mean(axis=0)

    d = center1 - center2
    theta = math.atan2(d[2], d[0])  # range -pi to pi
    distance = (d[2]**2 + d[0]**2)**0.5  # center distance on the ground

    box1 = corners1[[0, 1, 4, 5], :][:, [0, 2]]  # 4 corners of the bottom face (0&5, 1&4 are opposite corners)
    box2 = corners2[[0, 1, 4, 5], :][:, [0, 2]]

    # Note that bounding boxes might not be axis-aligned
    polygon1, polygon2 = Polygon(box1[[0, 1, 3, 2], :]), Polygon(box2[[0, 1, 3, 2], :])  # change the order to be convex
    point1, point2 = Point(center1[[0, 2]]), Point(center2[[0, 2]])

    # Initialize the relationship
    p = None

    # Horizontal relationship: "left"/"right"/"front"/"behind"
    if theta >= 3 * math.pi / 4 or theta < -3 * math.pi / 4:
        p = "left of"
    elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = "behind"
    elif -math.pi / 4 <= theta < math.pi / 4:
        p = "right of"
    elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = "in front of"

    # Vertical relationship: "above"/"below"
    if point1.within(polygon2) or point2.within(polygon1):
        delta1 = center1[1] - center2[1]
        delta2 = (
            corners1[:, 1].max() - corners1[:, 1].min() +
            corners2[:, 1].max() - corners2[:, 1].min()
        ) / 2.
        if (delta1 - delta2) >= 0. or "lamp" in name1:
            # Indicate that:
            # (1) delta1 > 0. (because always delta2 > 0.): `center1` is above `center2`
            # (2) delta1 >= delta2: `corners1` and `corners2` not intersect vertically
            # ==> `corners1` is completely above `corners2`
            # Or the subject is a lamp, which is always above other objects
            p = "above"
            return p, distance
        if (-delta1 - delta2) >= 0. or "lamp" in name2:
            # ==> `corners1` is completely below `corners2`
            # Or the object is a lamp, which is always above other objects
            p = "below"
            return p, distance

    if distance > 3.:
        return None, distance  # too far away
    else:
        if distance < 1.:
            p = "closely " + p
        return p, distance

def two_rectangle_collision(rect1, rect2):
    '''using Separating Axis Theorem''' 
    # input format: corner coordinate arrays
    def get_corners(obj):
        corners = []
        cx, cz = obj.position[0], obj.position[2]
        dx, dz = obj.size[0], obj.size[2]
        cos_angle, sin_angle = np.cos(obj.z_angle), np.sin(obj.z_angle) # TODO: verify coordinate frame
        for x, z in [(-dx, -dz), (dx, -dz), (dx, dz), (-dx, dz)]:
            corners.append((cx + x * cos_angle + z * sin_angle, cz - x * sin_angle + z * cos_angle))
        return np.array(corners)
    def project_rectangle_on_axis(rectangle, axis):
        """ Project a rectangle on an axis and return the minimum and maximum projections. """
        corners = get_corners(rectangle)
        projections = [corner[0] * axis[0] + corner[1] * axis[1] for corner in corners]
        return min(projections), max(projections)

    # Get the axes to test against
    axes = []
    for rect in (rect1, rect2):
        corners = get_corners(rect)
        # For each rectangle, the axes to test are the normals to its edges
        for i in range(4):
            edge = corners[i] - corners[i - 1]
            normal = (-edge[1], edge[0])
            axes.append(normal)
    
    # Test projections on all axes
    for axis in axes:
        min1, max1 = project_rectangle_on_axis(rect1, axis)
        min2, max2 = project_rectangle_on_axis(rect2, axis)
        # If there is no overlap on any axis, the rectangles do not collide
        if max1 < min2 or max2 < min1:
            return False
    
    # check whether their heights overlap
    min1, max1 = rect1.position[1] - rect1.size[1] / 2, rect1.position[1] + rect1.size[1] / 2
    if min1 < 0:
        min1, max1 = 0, max1 - min1
    min2, max2 = rect2.position[1] - rect2.size[1] / 2, rect2.position[1] + rect2.size[1] / 2
    if min2 < 0:
        min2, max2 = 0, max2 - min2
    if max1 < min2 or max2 < min1:
        return False
    else:
        return True

def check_collision(scene, obj_i):
    '''
    check whether obj_i is in collision with the others in the scene
    '''
    obj = scene.bboxes[obj_i]
    for i in range(len(scene.bboxes)):
        if i == obj_i:
            continue
        if two_rectangle_collision(scene.bboxes[i], obj):
            return True
    return False

def check_collision_all(scene):
    '''check collision between every pair of objects in the scene'''
    for i in range(len(scene.bboxes)):
        for j in range(i + 1, len(scene.bboxes)):
            if two_rectangle_collision(scene.bboxes[i], scene.bboxes[j]):
                return True
    return False

def check_obj_uniqueness(scene, obj):
    return sum([1 if o.model_jid == obj.model_jid  else 0 for o in scene.bboxes]) == 1

def obj_description_jid(scene, obj):
    if not check_obj_uniqueness(scene, obj):
        return f'one of [JID]{obj.model_jid}[/JID]'
    else:
        return f'[JID]{obj.model_jid}[/JID]'

def render_scene(desc, scene):
    render_savepath = '/data1/zhengkz/3D_datasets/3D-Future/preprocess/all_renders/render_scenes/debug_multiedit'
    os.system(f'mkdir -p {render_savepath}')
    try:
        cv2.imwrite(f'{render_savepath}/{desc}topview.png', scene.render(elevation=-89))
        cv2.imwrite(f'{render_savepath}/{desc}normalview.png', scene.render(elevation=-45))
    except:
        pass

def object_replace(scene, id, obj_i, obj_dictbylabel, render=False, trytimes=10, relation_dicts=None):
    '''
    replace object i to a new one in category from threefuture dataset
    '''
    mscene = deepcopy(scene)
    tmpscene = deepcopy(mscene)
    obj = mscene.bboxes[obj_i]
    obj_class = obj.label
    obj_class_str = deepcopy(obj_class).replace('/','-')
    n_try = 0
    success = False
    while n_try < trytimes:
        newobj = deepcopy(np.random.choice(obj_dictbylabel[obj_class]))
        if obj.model_jid != newobj.model_jid:
            # newobj.size = obj.size
            newobj.rotation = obj.rotation
            newobj.position = obj.position
            tmpscene.bboxes[obj_i] = newobj
            if not check_collision(tmpscene, obj_i):
                mscene.bboxes[obj_i] = newobj
                success = True
                break
            n_try += 1
    obj_des = obj_description_jid(scene, obj)
    mscene.command = f'replace source with target : [Source] {obj_des}; [Target] [JID]{newobj.model_jid}[/JID]'
    if "one of" in obj_des and relation_dicts is not None and obj_i in relation_dicts:
        mscene.command = mscene.command.replace('one of', '')
        relative_jid, relation_str, distance = relation_dicts[obj_i]
        mscene.command += f'; location: ***{relation_str}*** [JID]{relative_jid}[/JID]'
    mscene.original_id = id
    if success and render:
        render_scene(f'{id:06d}-replace-{obj_class_str}-', scene)
    return success, mscene
    
def pose_change(scene, id, obj_i, mode=0, render=False, relation_dicts=None, fixed_rotate=False, fixed_scale=False):
    '''
    scale/translate/rotate object by a certain amount, avoid collision if possible, otherwise find cases only collide with one object and move away that object and repeat process on that object

    mode=0: scale; mode=1: translate; mode=2: rotate.

    descriptions:
    scale: 'enlarge/shrink object by xx times'
    translate: 'move object along X/Z axis xx distance
    rotate: 'rotate object around Y axis xx degrees'
    '''
    mscene = deepcopy(scene)
    obj = mscene.bboxes[obj_i]
    obj_class = obj.model_info.super_category
    obj_class_str = deepcopy(obj_class).replace('/','-')
    success = None
    obj_des = obj_description_jid(scene, obj)
    if "one of" in obj_des and relation_dicts is not None and obj_i in relation_dicts:
        obj_des = obj_des.replace('one of', '')
        relative_jid, relation_str, distance = relation_dicts[obj_i]
        obj_des += f'; location: ***{relation_str}*** [JID]{relative_jid}[/JID]'
    des = ''
    if mode == 0:
        '''logic: for shrinking, it's always collision-free; for enlarging, check collision and skip if it is free; otherwise, linearly try smaller scale'''
        if fixed_scale:
            scale_factor = np.random.choice(np.array([0.8, 1.2, 0.5, 2.0]))
        else:
            scale_factor = np.random.choice([np.random.randint(5, 9)/10, np.random.randint(11, 15)/10])
        newobj = deepcopy(obj)
        newobj.size = scale_factor * obj.size
        newobj.scale = scale_factor * np.array(obj.scale)
        tmpscene = deepcopy(mscene)
        tmpscene.bboxes[obj_i] = newobj
        if scale_factor > 1:
            if not check_collision(tmpscene, obj_i):
                success = True # success
            else:
                factors = np.linspace(scale_factor, 1.1, 5)
                for scale_factor in factors:
                    if not check_collision(tmpscene, obj_i):
                        success = True
                        break # success, found a smaller enlarge scale
                # fail, cannot enlarge a bit
                if success is None:
                    # print(f'cannot enlarge {obj_class_str} even by 1.1x, always collision')
                    success = False
            if success:
                des = f'enlarge object by {scale_factor:.1f} X : {obj_des} \n'
                if scale_factor > 1.3:
                    semantic_des = f'obviously enlarge object :'
                else:
                    semantic_des = f'enlarge object :'
                semantic_des += obj_des
                des += semantic_des
                mscene.bboxes[obj_i] = newobj
        else:
            success = True
            des = f'shrink object by {scale_factor:.1f} X : {obj_des} \n'
            if scale_factor < 0.7:
                semantic_des = f'obviously shrink object :'
            else:
                semantic_des = f'shrink object :'
            semantic_des += obj_des
            des += semantic_des
            mscene.bboxes[obj_i] = newobj
        change_number = scale_factor
    elif mode == 1:
        '''logic: try meters + some random distances translation along x and z directions, if all in collision skip'''
        tmpscene = deepcopy(scene)
        newobj = deepcopy(obj)
        tmpscene.bboxes[obj_i] = newobj
        trans_dists = np.random.permutation(np.concatenate((np.random.uniform(0.5, 2.5, 10), np.linspace(0.5, 2.45, 15))))
        directions = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]] # x+, x-, z+, z-
        all_dir_des = ['right', 'left', 'front', 'back']
        for dist in trans_dists:
            directions_id = np.random.permutation(np.arange(4))
            for idx in directions_id:
                direction = directions[idx]
                dir_des = all_dir_des[idx]
                newobj.position = (np.array(obj.position) + np.array(direction) * dist).tolist()
                if not check_collision(tmpscene, obj_i):
                    success = True
                    mscene.bboxes[obj_i] = newobj
                    break
            if success is True:
                break
        if success is None:
            success = False
            # print(f'cannot move {obj_class_str} even between 0.05-2, always collision')
        des = f'move object towards the ***{dir_des}*** direction for {dist:.2f} meters: {obj_des}\n'
        if dist > 1:
            semantic_des = f'obviously move object towards the ***{dir_des}*** direction :'
        elif dist < 0.5:
            semantic_des = f'slightly move object towards the ***{dir_des}*** direction :'
        else:
            semantic_des = f'move object towards the ***{dir_des}*** direction :'
        semantic_des += obj_des
        des += semantic_des
        change_number = dist
    elif mode == 2:
        '''logic: try 15, 30, 45, ..., 165 degrees + some random degrees rotation, if all in collision try to shrink a bit like 0.8, otherwise fail'''
        tmpscene = deepcopy(scene)
        newobj = deepcopy(obj)
        tmpscene.bboxes[obj_i] = newobj
        if fixed_rotate:
            rot_angles = np.random.permutation(np.array([90, -90, 180]))
        else:
            rot_angles = np.random.permutation(np.concatenate((-np.linspace(15, 165, 11), np.linspace(15, 180, 12))))
        for angle in rot_angles:
            rotmat = R.from_quat(obj.rotation).as_matrix()
            newobj.rotation = R.from_matrix(R.from_rotvec([0, angle, 0], True).as_matrix() @ rotmat).as_quat()
            if not check_collision(tmpscene, obj_i):
                success = True
                mscene.bboxes[obj_i] = newobj
                break
        if success:
            des = f'rotate object {angle:.0f} degrees : {obj_des}\n'
            abs_angle = np.abs(angle)
            if abs_angle >= 135:
                semantic_des = f'obviously rotate object {abs_angle:.0f} degrees :'
            elif abs_angle <= 45:
                semantic_des = f'slightly rotate object {abs_angle:.0f} degrees :'
            else:
                semantic_des = f'rotate object {abs_angle:.0f} degrees :'
            if angle > 0:
                semantic_des.replace('rotate', 'counterclockwise rotate')
            semantic_des += obj_des
            des += semantic_des
        if success is None:
            success = False
        change_number = angle
    mscene.command = des
    mscene.original_id = id
    if render and success:
        render_scene(f'{id:06d}-{mode:d}-{des}-', mscene)
    return success, mscene, change_number

def add_remove(scene, id, obj_i, relation_dicts=None):
    '''
    add and remove object_i to create new scene
    '''
    mscene = deepcopy(scene)
    removed_scene = deepcopy(scene)
    # obj = removed_scene.bboxes[obj_i]
    obj = removed_scene.bboxes.pop(obj_i)

    obj_des = obj_description_jid(scene, obj)
    if relation_dicts is not None and obj_i in relation_dicts:
        relative_jid, relation_str, distance = relation_dicts[obj_i]
        obj_des += f'; location: ***{relation_str}*** [JID]{relative_jid}[/JID]'
    removed_scene.command = f'remove object: {obj_des}'

    mscene.command = f'add object: {obj_des}'

    return removed_scene, mscene

### OpenAI Utils ###
def submit_batch_files(batch_files, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    path_to_id = {}
    for file in batch_files:
        batch_input_file = client.files.create(
            file=open(file, "rb"),
            purpose="batch"
          )
        batch_input_file_id = batch_input_file.id
        request = client.batches.create(
              input_file_id=batch_input_file_id,
              endpoint="/v1/chat/completions",
              completion_window="24h",
              metadata={
                "source_file": file,
              }
          )
        path_to_id[file] = request.id
    return path_to_id

def retrieve_batch_response(path_to_id, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    file_to_content = {}
    for file, batch_id in path_to_id.items():
        response = client.batches.retrieve(batch_id)
        if response.status == "completed":
            print(f"Batch {file} succeeded.")
            content = client.files.content(response.output_file_id)
        else:
            print(f"Batch {file} is not completed. Status: {response.status}")
            content = None
        file_to_content[file] = content
    return file_to_content