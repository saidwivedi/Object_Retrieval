"""
Utility functions for object lookup functionality.
"""
import json
import requests
import trimesh
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
import transformers
from huggingface_hub import hf_hub_download
import objaverse

OBJAVERSE_LVIS_LABELS = objaverse.load_lvis_annotations()

def get_closest_labels_from_objaverse_lvis(user_label, model, threshold=0.75):

    labels = OBJAVERSE_LVIS_LABELS

    # Clean the input to be more compatible with the model's vocabulary
    user_input = user_label.lower().replace("_", " ")

    # Get the vector for the user's input. Handle cases where it's not in the vocabulary.
    try:
        input_vector = model[user_input]
    except KeyError:
        # Check if any part of a multi-word input is in the vocabulary
        input_words = user_input.split()
        input_vectors = [model[word] for word in input_words if word in model]
        if not input_vectors:
            return f"Sorry, '{user_input}' is not in the vocabulary."
        input_vector = np.mean(input_vectors, axis=0)


    similarities = []
    for original_label in labels:
        # Clean the label for processing
        cleaned_label = original_label.lower().replace("_", " ").replace('_(', ' (').replace(')', '')
        
        label_words = cleaned_label.split()
        label_vectors = [model[word] for word in label_words if word in model]
        
        if label_vectors:
            # Average the vectors for multi-word labels
            label_vector = np.mean(label_vectors, axis=0)
            
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(input_vector, label_vector)
            
            if similarity >= threshold:
                similarities.append((original_label, similarity))

    # Sort the results by similarity in descending order for better readability
    similarities.sort(key=lambda item: item[1], reverse=True)
    labels = None if not similarities else [label for label, _ in similarities]

    return similarities, labels

def get_objaverse_objs():
    """Load precomputed Objaverse embeddings and metadata."""
    meta = json.load(open(hf_hub_download("OpenShape/openshape-objaverse-embeddings", 
                         "objaverse_meta.json", repo_type='dataset')))

    lvis_meta = {x['u']: x for x in meta['entries']}
    deser = torch.load(
        hf_hub_download("OpenShape/openshape-objaverse-embeddings", "objaverse.pt", repo_type='dataset'), map_location='cpu'
    )
    lvis_uids = deser['us']
    lvis_feats = deser['feats']

    print(f'Loaded {len(lvis_meta)} objects with precomputed features')

    return lvis_meta, lvis_feats, lvis_uids

def get_objaverse_lvis_objs():

    lvis_metadata = json.load(open('assets/lvis.json'))
    lvis_objs_cat = {}
    for sample in lvis_metadata:
        s_cat = sample['category'].lower()
        s_uid = sample['uid']

        if s_cat not in lvis_objs_cat:
            lvis_objs_cat[s_cat] = {}
            lvis_objs_cat[s_cat]['uid'] = []
        
        lvis_objs_cat[s_cat]['uid'].append(s_uid)

    return lvis_objs_cat

def get_objaverse_lvis_objs_subset(valid_labels_list, lvis_objs_cat, objaverse_meta):

    uid_list, subset_meta = [], {}
    for label in valid_labels_list:
        if label in lvis_objs_cat.keys():
            uid_list.extend(lvis_objs_cat[label]['uid'])
    
    for uid in uid_list:
        subset_meta.update({uid: objaverse_meta[uid]})

    return subset_meta


def load_openclip():
    """Load OpenCLIP model and processor."""
    half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    clip_model, clip_prep = transformers.CLIPModel.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        low_cpu_mem_usage=True, torch_dtype=half,
        offload_state_dict=True
    ), transformers.CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    if torch.cuda.is_available():
        clip_model.cuda()
    return clip_model, clip_prep


def download_glb(url, glb_path='glb_mesh.glb'):
    """Download GLB file from URL."""
    response = requests.get(url)
    with open(glb_path, 'wb') as file:
        file.write(response.content)


def convert_glb_to_obj(glb_path='glb_mesh.glb', obj_path='converted_mesh.obj'):
    """Convert GLB file to OBJ format."""
    # Load the GLB file using trimesh
    scene = trimesh.load(glb_path)

    if not isinstance(scene, trimesh.Scene):
        raise ValueError("Invalid GLB file. Unable to convert.")

    # Concatenate all meshes into a single mesh
    mesh = trimesh.util.concatenate([scene.dump(concatenate=True)])

    # Save the merged mesh to an OBJ file
    try:
        if isinstance(mesh, list):
            print(mesh)
            mesh = mesh[0]
        mesh.export(obj_path, include_color=False, include_texture=False, write_texture=False)
        print(f"Conversion successful. OBJ file saved at: {obj_path}")
    except:
        print(f'Failed in converting')


def retrieval_filter_expand(sim_th=0.05):
    """Create filtering function for retrieval results."""
    tag = "" #tag if any
    face_min, face_max = 0, 34985808
    anim_min, anim_max = 0, 563
    tag_n = not bool(tag.strip())
    anim_n = not (anim_min > 0 or anim_max < 563)
    face_n = not (face_min > 0 or face_max < 34985808)
    filter_fn = lambda x: (
        (anim_n or anim_min <= x['anims'] <= anim_max)
        and (face_n or face_min <= x['faces'] <= face_max)
        and (tag_n or tag in x['tags'])
    )
    return sim_th, filter_fn
    

def retrieve(embedding, top, lvis_meta, lvis_feats, lvis_uids, sim_th=0.0, filter_fn=None):
    """Retrieve similar objects from precomputed embeddings."""
    sims = []
    embedding = F.normalize(embedding.detach().cpu(), dim=-1).squeeze()
    for chunk in torch.split(lvis_feats, 10240):
        sims.append(embedding @ F.normalize(chunk.float(), dim=-1).T)
    sims = torch.cat(sims)
    sims, idx = torch.sort(sims, descending=True)
    sim_mask = sims > sim_th
    sims = sims[sim_mask]
    idx = idx[sim_mask]
    results = []
    for i, sim in zip(idx, sims):
        if lvis_uids[i] in lvis_meta:
            if filter_fn is None or filter_fn(lvis_meta[lvis_uids[i]]):
                results.append(dict(lvis_meta[lvis_uids[i]], sim=sim))
                if len(results) >= top:
                    break
    return results
