import os

from datasets.icfgpedes import ICFGPEDES

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.metrics import Evaluator
from utils.iotools import load_train_configs
import random
import matplotlib.pyplot as plt
from PIL import Image

config_file = 'logs/ICFG-PEDES/sdm+itc+aux_lr3e-06_test/configs.yaml'
args = load_train_configs(config_file)
args.batch_size = 64
args.training = False
device = "cuda"
test_img_loader, test_txt_loader, num_class = build_dataloader(args)
model = build_model(args, num_class)
checkpointer = Checkpointer(model)
checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
model.to(device)

evaluator = Evaluator(test_img_loader, test_txt_loader)

qfeats, gfeats, qids, gids, _, _ = evaluator._compute_embedding(model.eval())
qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

similarity = qfeats @ gfeats.t()
_, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)
indices = indices.cpu()
qfeats = qfeats.cpu()
gfeats = gfeats.cpu()

# Load dataset to get correct gt_img_paths per caption
dataset = ICFGPEDES(root='/media/jqzhu/哈斯提·基拉/UniMoESE/data')
test_dataset = dataset.test

img_paths = test_dataset['img_paths']  # gallery image paths (indexed by gids)
captions = test_dataset['captions']  # all query captions
gt_img_paths = test_dataset['gt_img_paths']  # ✅ now correctly added: one per caption


def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths, gt_img_paths):
    query_caption = captions[idx]
    query_id = qids[idx]
    retrieved_img_paths = [img_paths[j] for j in indices[idx]]
    retrieved_img_ids = gids[indices[idx]]
    gt_img_path = gt_img_paths[idx]  # ✅ this is now the true image that the caption describes
    return query_id, retrieved_img_ids, query_caption, retrieved_img_paths, gt_img_path


import os

from datasets.icfgpedes import ICFGPEDES

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.metrics import Evaluator
from utils.iotools import load_train_configs
import random
import matplotlib.pyplot as plt
from PIL import Image

config_file = 'logs/ICFG-PEDES/sdm+itc+aux_lr3e-06_test/configs.yaml'
args = load_train_configs(config_file)
args.batch_size = 64
args.training = False
device = "cuda"
test_img_loader, test_txt_loader, num_class = build_dataloader(args)
model = build_model(args, num_class)
checkpointer = Checkpointer(model)
checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
model.to(device)

evaluator = Evaluator(test_img_loader, test_txt_loader)

qfeats, gfeats, qids, gids, _, _ = evaluator._compute_embedding(model.eval())
qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

similarity = qfeats @ gfeats.t()
_, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)
indices = indices.cpu()
qfeats = qfeats.cpu()
gfeats = gfeats.cpu()

# Load dataset to get correct gt_img_paths per caption
dataset = ICFGPEDES(root='/media/jqzhu/哈斯提·基拉/UniMoESE/data')
test_dataset = dataset.test

img_paths = test_dataset['img_paths']  # gallery image paths (indexed by gids)
captions = test_dataset['captions']  # all query captions
gt_img_paths = test_dataset['gt_img_paths']  # ✅ now correctly added: one per caption


def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths, gt_img_paths):
    query_caption = captions[idx]
    query_id = qids[idx]
    retrieved_img_paths = [img_paths[j] for j in indices[idx]]
    retrieved_img_ids = gids[indices[idx]]
    gt_img_path = gt_img_paths[idx]  # ✅ this is now the true image that the caption describes
    return query_id, retrieved_img_ids, query_caption, retrieved_img_paths, gt_img_path



import os
from PIL import Image
import matplotlib.pyplot as plt

def plot_retrieval_images_and_save_text(indices, qids, gids, captions, img_paths, gt_img_paths, save_dir, ids_batch=None):
    """
    This function takes a batch of query ids, retrieves corresponding captions,
    and generates images showing the ground truth and retrieved images for each query.
    Saves both the images and the captions as text files in the specified folder.

    :param indices: The indices of the top-k retrieved images.
    :param qids: The query IDs.
    :param gids: The gallery IDs.
    :param captions: The query captions.
    :param img_paths: Paths to the gallery images.
    :param gt_img_paths: The ground truth image paths.
    :param save_dir: The directory to save the images and text files.
    :param ids_batch: List of query IDs to visualize (if None, uses all queries).
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Loop through the given query IDs batch
    for idx in (ids_batch if ids_batch else range(len(qids))):
        query_caption = captions[idx]
        query_id = qids[idx]
        retrieved_img_paths = [img_paths[j] for j in indices[idx]]
        retrieved_img_ids = gids[indices[idx]]
        gt_img_path = gt_img_paths[idx]  # Ground truth image

        # Plot the retrieval images
        fig = plt.figure(figsize=(16, 3))
        col = len(retrieved_img_paths)

        # Plot ground truth image (leftmost)
        plt.subplot(1, col + 1, 1)
        img = Image.open(gt_img_path).convert('RGB')
        img = img.resize((128, 256))
        plt.imshow(img)
        plt.title("Ground Truth", fontsize=8)
        plt.xticks([]); plt.yticks([])

        # Plot retrieved images
        for i in range(col):
            plt.subplot(1, col + 1, i + 2)
            img = Image.open(retrieved_img_paths[i]).convert('RGB')
            img = img.resize((128, 256))
            plt.imshow(img)
            plt.title(f"Rank {i + 1}", fontsize=8)
            plt.xticks([]); plt.yticks([])

            ax = plt.gca()
            if retrieved_img_ids[i] == query_id:
                color = 'lawngreen'
            else:
                color = 'red'
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(3)

        # Save the image to the specified folder
        plt.tight_layout()
        img_name = f"query_{query_id}_retrievals.png"
        img_path = os.path.join(save_dir, img_name)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        # Save the query caption as a text file
        caption_filename = f"query_{query_id}_caption.txt"
        caption_path = os.path.join(save_dir, caption_filename)
        with open(caption_path, 'w') as f:
            f.write(query_caption)

# Example usage:
# Assuming `indices`, `qids`, `gids`, `captions`, `img_paths`, `gt_img_paths` are already defined
ids_batch = [208,3306,4000,312,104,2,35,1000]  # List of query IDs to visualize
save_dir = 'retrieval_results'  # Folder to save the images and text files

plot_retrieval_images_and_save_text(indices, qids, gids, captions, img_paths, gt_img_paths, save_dir, ids_batch)
