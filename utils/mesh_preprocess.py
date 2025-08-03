import os
import random

import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
import trimesh


def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

def model_to_pc(mesh: trimesh.Trimesh, n_sample_points=10000):
    f32 = np.float32
    rad = np.sqrt(mesh.area / (3 * n_sample_points))
    for _ in range(24):
        pcd, face_idx = trimesh.sample.sample_surface_even(mesh, n_sample_points, rad)
        rad *= 0.85
        if len(pcd) == n_sample_points:
            break
    else:
        raise ValueError("Bad geometry, cannot finish sampling.", mesh.area)
    if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
        rgba = mesh.visual.face_colors[face_idx]
    elif isinstance(mesh.visual, trimesh.visual.TextureVisuals):
        bc = trimesh.proximity.points_to_barycentric(mesh.triangles[face_idx], pcd)
        if mesh.visual.uv is None or len(mesh.visual.uv) < mesh.faces[face_idx].max():
            uv = np.zeros([len(bc), 2])
            # print("Invalid UV, filling with zeroes")
        else:
            uv = np.einsum('ntc,nt->nc', mesh.visual.uv[mesh.faces[face_idx]], bc)
        material = mesh.visual.material
        if hasattr(material, 'materials'):
            if len(material.materials) == 0:
                rgba = np.ones_like(pcd) * 0.8
                texture = None
                print("Empty MultiMaterial found, falling back to light grey")
            else:
                material = material.materials[0]
        if hasattr(material, 'image'):
            texture = material.image
            if texture is None:
                rgba = np.zeros([len(uv), len(material.main_color)]) + material.main_color
        elif hasattr(material, 'baseColorTexture'):
            texture = material.baseColorTexture
            if texture is None:
                rgba = np.zeros([len(uv), len(material.main_color)]) + material.main_color
        else:
            texture = None
            rgba = np.ones_like(pcd) * 0.8
            print("Unknown material, falling back to light grey")
        if texture is not None:
            rgba = trimesh.visual.uv_to_interpolated_color(uv, texture)
    if rgba.max() > 1:
        if rgba.max() > 255:
            rgba = rgba.astype(f32) / rgba.max()
        else:
            rgba = rgba.astype(f32) / 255.0
    return np.concatenate([np.array(pcd, f32), np.array(rgba, f32)[:, :3]], axis=-1)

def trimesh_to_pc(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = []
        for node_name in scene_or_mesh.graph.nodes_geometry:
            # which geometry does this node refer to
            transform, geometry_name = scene_or_mesh.graph[node_name]

            # get the actual potential mesh instance
            geometry = scene_or_mesh.geometry[geometry_name].copy()
            if not hasattr(geometry, 'triangles'):
                continue
            geometry: trimesh.Trimesh
            geometry = geometry.apply_transform(transform)
            meshes.append(geometry)
        total_area = sum(geometry.area for geometry in meshes)
        if total_area < 1e-6:
            raise ValueError("Bad geometry: total area too small (< 1e-6)")
        pcs = []
        for geometry in meshes:
            pcs.append(model_to_pc(geometry, max(1, round(geometry.area / total_area * 10000))))
        if not len(pcs):
            raise ValueError("Unsupported mesh object: no triangles found")
        return np.concatenate(pcs)
    else:
        return model_to_pc(scene_or_mesh, 10000)

def _load_data_as_pc(obj, num_points=10000, y_up=True, verbose=True):
    if isinstance(obj, str):
        file_name = obj
        if file_name.endswith('.obj'):
            mesh = trimesh.load(file_name, process=False)
            pc = trimesh_to_pc(mesh)
            xyz = np.asarray(pc[:, :3])
            rgb = np.asarray(pc[:, 3:])
        elif file_name.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(file_name)
            xyz = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)
    elif isinstance(obj, dict):
        obj = list(obj.values())[0]
        if obj.shape[1] == 6:
            xyz = obj[:, :3]
            rgb = obj[:, 3:]
        else:
            xyz = obj
            rgb = np.ones_like(xyz) * 0.8

    n = xyz.shape[0]
    if n != num_points:
        if verbose:
            print(f'Number of points in the point cloud is {n}, sampling {num_points} points')
        idx = random.sample(range(n), num_points)
        xyz = xyz[idx]
        rgb = rgb[idx]
    if y_up:
        # swap y and z axis
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
    xyz = normalize_pc(xyz)
    if rgb is None:
        rgb = np.ones_like(rgb) * 0.4
    features = np.concatenate([xyz, rgb], axis=1)
    xyz = torch.from_numpy(xyz).type(torch.float32)
    features = torch.from_numpy(features).type(torch.float32)

    return xyz, features

def get_shape_features(model, objlist, num_points=10000, y_up=True, model_arch='spconv', device='cuda'):

    # supported object types: .obj, .ply, dict with keys as file name and values as point cloud

    uids_batch = []
    processed_feats = []

    for idx, obj in enumerate(objlist):
        if isinstance(obj, dict):
            fname = list(obj.keys())[0]
        elif isinstance(obj, str):
            fname = obj
        else:
            raise ValueError('Invalid object type')

        xyz, feat = _load_data_as_pc(obj, num_points, y_up)
        uids_batch.append(fname)
        if 'pointbert' in model_arch:
            xyz = torch.stack([xyz]).float().to(device)
            feat = torch.stack([feat]).float().to(device)
        elif 'spconv' in model_arch:
            xyz = ME.utils.batched_coordinates([xyz], dtype=torch.float32).to(device)
            feat = torch.cat([feat], dim=0).to(device)
        with torch.no_grad():
            shape_features = model(xyz, feat, device='cuda', quantization_size=0.02)
        processed_feats.append(shape_features[0])

    processed_feats = torch.stack(processed_feats)
    return {
        'shape_feats': processed_feats,
        'uids': uids_batch,
    }