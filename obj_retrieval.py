import os
from os.path import join
import torch
import argparse
from PIL import Image
import torch.nn.functional as F

from models.model import load_model, get_model_name
import gensim.downloader as api
from utils.mesh_preprocess import get_shape_features
from utils.utils import (
    load_openclip, 
    download_glb, 
    convert_glb_to_obj, 
    get_objaverse_objs,
    get_objaverse_lvis_objs,
    get_objaverse_lvis_objs_subset,
    get_closest_labels_from_objaverse_lvis,
    retrieval_filter_expand,
    retrieve
)


def do_lookup(objlist, imglist, model_type='objaverse', model_name=None, top_k=2, y_gravity_axis=True):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    
    model_name, model_arch = get_model_name(model_name=model_name, model_type=model_type)
    model = load_model(model_name).to(device)
    clip_model, clip_prep = load_openclip()

    # get shape features
    print(f'\n-->Getting shape features for objects')
    out_dict = get_shape_features(model, objlist, model_arch=model_arch, y_up=y_gravity_axis)
    shape_feat, obj_names = out_dict['shape_feats'], out_dict['uids']
    print(f'Object Shape features: {shape_feat.shape}')

    # get image features
    print(f'\n-->Getting image features for images')
    images = [Image.open(f).convert("RGB") for f in imglist]
    tn = clip_prep(images=images, return_tensors="pt").to(device)
    image_feat = clip_model.get_image_features(pixel_values=tn['pixel_values'].type(half)).float()
    print(f'Image features: {image_feat.shape}')

    # get similarity scores
    sim_scores = F.normalize(shape_feat, dim=1) @ F.normalize(image_feat, dim=1).T
    sim_scores = sim_scores.cpu().detach().numpy()

    lookup_results = {}
    for i, img in enumerate(imglist):
        mesh_index = sim_scores[:, i].argsort()[::-1][:top_k]
        most_similar_meshes = [obj_names[idx] for idx in mesh_index]
        lookup_results[img] = {
            'most_similar_meshes': most_similar_meshes,
            'similarity_scores': sim_scores[mesh_index, i]
        }

    return lookup_results

def do_precomputed_lookup(imglist, top_k=8, sim_th=0.05, download_objects=False, output_dir=None, use_label_filtering=True):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    
    print('\n-->Loading precomputed Objaverse data...')
    objaverse_meta, objaverse_feats, objaverse_uids = get_objaverse_objs()
    
    if use_label_filtering:
        lvis_objs_cat = get_objaverse_lvis_objs()
        print(f'Loading word2vec model for label matching...')
        word2vec_model = api.load("fasttext-wiki-news-subwords-300")
    else:
        lvis_objs_cat = None
        word2vec_model = None
        print('Label-based filtering disabled - using entire Objaverse database')

    clip_model, clip_prep = load_openclip()
    sim_th, filter_fn = retrieval_filter_expand(sim_th=sim_th)


    lookup_results = {}
    
    for imgpath in imglist:
        imgname = os.path.basename(imgpath)
        print(f'\n-->Processing image: {imgname}')
        
        img = Image.open(imgpath).convert("RGB")

        # Apply label-based filtering if enabled
        if use_label_filtering:
            obj_cat = imgname.split('_')[0] if '__' in imgname else None
            if obj_cat:
                similarities, closest_labels = get_closest_labels_from_objaverse_lvis(obj_cat, word2vec_model, threshold=0.75)
                print(f'Closest labels for category "{obj_cat}": {similarities}')
                matched_objs_meta = get_objaverse_lvis_objs_subset(closest_labels, lvis_objs_cat, objaverse_meta)
                print(f'Using {len(matched_objs_meta)} objects for category "{obj_cat}"')
            else:
                matched_objs_meta = objaverse_meta
                print(f'No category detected in filename, using full Objaverse database ({len(objaverse_meta)} objects)')
        else:
            matched_objs_meta = objaverse_meta
            print(f'Using entire Objaverse database ({len(objaverse_meta)} objects)')

        # Get image features
        tn = clip_prep(images=[img], return_tensors="pt").to(device)
        image_feat = clip_model.get_image_features(pixel_values=tn['pixel_values'].type(half)).float().cpu()
        
        # Retrieve similar objects
        results = retrieve(image_feat, top_k, matched_objs_meta, objaverse_feats, objaverse_uids, sim_th, filter_fn)

        if len(results) == 0:
            print(f'No results found for {imgname}')
            lookup_results[imgpath] = {
                'most_similar_objects': [],
                'similarity_scores': [],
                'metadata': [],
                'download_urls': []
            }
            continue
        
        # Extract results
        similar_objects = []
        similarity_scores = []
        metadata = []
        download_urls = []
        
        for result in results:
            similar_objects.append(result['u'])  # UID
            similarity_scores.append(float(result['sim']))
            metadata.append(result)
            glb_file = result['glb']
            download_url = f'https://huggingface.co/datasets/allenai/objaverse/resolve/main/{glb_file}'
            download_urls.append(download_url)
        
        lookup_results[imgpath] = {
            'most_similar_objects': similar_objects,
            'similarity_scores': similarity_scores,
            'metadata': metadata,
            'download_urls': download_urls
        }
        
        print(f'Found {len(results)} similar objects for {imgname}')
        print(f'Top result: {results[0]["u"]} (similarity: {results[0]["sim"]:.4f})')
        
        # Display link to view the top matched object
        top_glb_file = results[0]['glb']
        display_url = f'https://huggingface.co/datasets/allenai/objaverse/blob/main/{top_glb_file}'
        print(f'View top result: {display_url}')
        
        # Optionally download and convert objects
        if download_objects and output_dir and len(results) > 0:
            os.makedirs(output_dir, exist_ok=True)
            
            # Download and convert the top result
            top_result = results[0]
            glb_file = top_result['glb']
            download_url = f'https://huggingface.co/datasets/allenai/objaverse/resolve/main/{glb_file}'
            
            try:
                glb_path = os.path.join(output_dir, f'{imgname[:-4]}_top_result.glb')
                obj_path = os.path.join(output_dir, f'{imgname[:-4]}_top_result.obj')
                
                print(f'Downloading top result for {imgname}...')
                download_glb(download_url, glb_path)
                convert_glb_to_obj(glb_path, obj_path)
                
                # Clean up GLB file
                if os.path.exists(glb_path):
                    os.remove(glb_path)
                    
            except Exception as e:
                print(f'Failed to download/convert object for {imgname}: {e}')

    return lookup_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='objaverse', help='Choose model type: shapenet or objaverse')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--obj_folder', type=str, default='./assets/objs', help='object meshes folder')
    parser.add_argument('--img_folder', type=str, default='./assets/imgs', help='image folder')
    parser.add_argument('--top_k', type=int, default=2, help='top k results')
    parser.add_argument('--y_gravity_axis', type=bool, default=True, help='y axis is gravity axis')
    
    # New arguments for precomputed lookup
    parser.add_argument('--use_precomputed', action='store_true', help='Use precomputed Objaverse embeddings instead of local objects')
    parser.add_argument('--sim_threshold', type=float, default=0.05, help='Similarity threshold for precomputed lookup')
    parser.add_argument('--download_objects', action='store_true', help='Download and convert top results to OBJ files')
    parser.add_argument('--output_dir', type=str, default='./downloaded_objects', help='Directory to save downloaded OBJ files')
    parser.add_argument('--use_label_filtering', action='store_true', default=False, help='Use label-based filtering from LVIS subset (default: True)')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    
    imglist = [join(args.img_folder, f) for f in sorted(os.listdir(args.img_folder)) if f.endswith('.png') or f.endswith('.jpg')]
    
    if args.use_precomputed:
        print("Using precomputed Objaverse embeddings for object lookup...")
        if args.use_label_filtering:
            print("Label-based filtering enabled - will use LVIS subset when category is detected in filename")
        else:
            print("Label-based filtering disabled - searching entire Objaverse database")
            
        lookup_results = do_precomputed_lookup(
            imglist=imglist,
            top_k=args.top_k,
            sim_th=args.sim_threshold,
            download_objects=args.download_objects,
            output_dir=args.output_dir if args.download_objects else None,
            use_label_filtering=args.use_label_filtering
        )
        
    else:
        print("Using local object files for lookup...")
        objlist = [os.path.join(args.obj_folder, f) for f in sorted(os.listdir(args.obj_folder)) if f.endswith('.obj')]
        
        lookup_results = do_lookup(objlist=objlist, \
                                   imglist=imglist, \
                                   model_type=args.model_type, \
                                   model_name=args.model_name, \
                                   top_k=args.top_k, \
                                   y_gravity_axis=args.y_gravity_axis)
        
        # Display results
        for img, res in lookup_results.items():
            print(f"For image {img}: \n\t Most similar meshes: {res['most_similar_meshes']} \n\t Similarity scores: {res['similarity_scores']}")




