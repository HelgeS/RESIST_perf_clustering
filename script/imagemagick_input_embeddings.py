import glob
import os

import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.models import ViT_B_16_Weights
from torchvision.models.vision_transformer import vit_b_16
from tqdm import tqdm

preprocessing = ViT_B_16_Weights.DEFAULT.transforms(antialias=True)


def encode(vit, image_path):
    img = read_image(image_path, mode=ImageReadMode.RGB)
    img = preprocessing(img)

    # Add batch dimension
    img = img.unsqueeze(0)

    feats = vit._process_input(img)

    # Expand the class token to the full batch
    batch_class_token = vit.class_token.expand(img.shape[0], -1, -1)
    feats = torch.cat([batch_class_token, feats], dim=1)

    feats = vit.encoder(feats)

    return feats[:, 0]


if __name__ == "__main__":
    data_dir = "./data/"
    imagenet_dir = "/home/helge/storage/datasets/imagenet"

    properties_file = os.path.join(data_dir, "imagemagick/others/properties.csv")
    new_properties_file = properties_file.replace(
        "properties.csv", "input_embeddings.csv"
    )

    vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()

    embeddings = []

    for s in tqdm(pd.read_csv(properties_file).name):
        paths = glob.glob(os.path.join(imagenet_dir, f"**/{s}.JPEG"), recursive=True)
        assert len(paths) == 1, f"Found multiple results for {s}: {paths}"

        enc = encode(vit, paths[0])
        embeddings.append([s] + enc[0].tolist())

    df = pd.DataFrame(
        embeddings, columns=["name"] + [f"v{i}" for i in range(enc[0].shape[-1])]
    )
    df.to_csv(new_properties_file, index_label="id")
