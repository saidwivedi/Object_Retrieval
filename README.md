# Object Retrieval from Single Image (Used in InteractVLM, CVPR2025)

A simple object retrieval tool designed as a component for [InteractVLM](https://github.com/your-username/interactvlm). This tool enables efficient object lookup and retrieval from a single image using both the large Objaverse database and local 3D object collections.

**Key Features**:
- **Precomputed Objaverse embeddings** for fast retrieval from 800K+ objects
- **Label-based filtering** using semantic similarity to improve accuracy for occluded/complex scenes
- **Local object collection** support for custom 3D mesh databases
- **Automatic object download** and conversion from Objaverse

This tool is built using [OpenShape](https://colin97.github.io/OpenShape/) (NeurIPS 2023).

## Installation

### Basic Installation
```bash
conda create -n object_lookup python=3.9
conda activate object_lookup
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install huggingface_hub transformers trimesh open3d tqdm einops
```

### For ShapeNet Models (Optional)
If you plan to use ShapeNet-trained models, install MinkowskiEngine:
```bash
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
conda install -c dglteam/label/cu113 dgl
```

### Additional Dependencies
```bash
pip install -r requirements.txt 
```

**Note**: For label-based object filtering functionality, the system automatically downloads FastText Word2Vec embeddings (`fasttext-wiki-news-subwords-300`) from Gensim on first use. This requires an internet connection and may take a few minutes for the initial download.

## Usage

### 1. Object Retrieval from Objaverse
Retrieve objects from the Objaverse database:

```bash
# Download top results from entire Objaverse database (800K Objects)
python obj_retrieval.py --use_precomputed \
                        --img_folder ./assets/imgs \
                        --download_objects \
                        --output_dir ./retrieved_objects

# Download top results from Objaverse-LVIS subset (55K Objects)
# Objaverse-LVIS is a manually annotated subset of Objaverse. However, some samples have multiple other objects or scenes.
python obj_retrieval.py --use_precomputed \
                        --use_label_filtering \
                        --img_folder ./assets/imgs \
                        --download_objects \
                        --output_dir ./retrieved_objects
```

#### Label-Based Object Filtering
For improved accuracy, especially when dealing with occluded objects or complex scenes, you can use object category labels in your image filenames. When an object label is detected in the filename and the flag `--use_label_filtering` is used, the system automatically filters the search to semantically similar categories using Word2Vec embeddings. 

**Note**: When the image contains only the target object with minimal background, running the lookup with the entire Objaverse database typically yields good results without requiring label filtering.

**Filename Convention**: Use double underscores (`__`) to separate the object category from other parts of the filename:

**Image Quality Recommendations**:
- **Cropped and segmented images** of the target object are **strongly encouraged** for best results
- **Full scene images** will work but may be less accurate even with label-based filtering since Objaverse has very noisy samples.

### 2. Object Retrieval from Your Own Collection of Object Meshes
Retrieve similar objects from your local mesh collection:

```bash
# Basic usage with local objects
python obj_retrieval.py --obj_folder ./assets/objs \
                        --img_folder ./assets/imgs \
                        --top_k 5
```

## Citation

If you use this utility in your research, please cite the original OpenShape paper and InteractVLM:

```bibtex
@misc{liu2023openshape,
      title={OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding}, 
      author={Minghua Liu and Ruoxi Shi and Kaiming Kuang and Yinhao Zhu and Xuanlin Li and Shizhong Han and Hong Cai and Fatih Porikli and Hao Su},
      year={2023},
      eprint={2305.10764},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@inproceedings{dwivedi_interactvlm_2025,
    title     = {{InteractVLM}: {3D} Interaction Reasoning from {2D} Foundational Models},
    author    = {Dwivedi, Sai Kumar and Antić, Dimitrije and Tripathi, Shashank and Taheri, Omid and Schmid, Cordelia and Black, Michael J. and Tzionas, Dimitrios},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```

## License

This project builds upon OpenShape and is intended for research and educational purposes. Please refer to the original [OpenShape repository](https://github.com/Colin97/OpenShape_code) for licensing details.