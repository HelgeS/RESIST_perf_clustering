import glob
from abc import ABC
from typing import Dict, List, Union
import os

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# The code to run starencoder is taken from the bigcode project
# URL: https://github.com/bigcode-project/bigcode-encoder

MASK_TOKEN = "<mask>"
SEPARATOR_TOKEN = "<sep>"
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"

DEVICE = "cpu"
MAX_INPUT_LEN = 10000
MAX_TOKEN_LEN = 1024


def set_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    output_data = {}
    for k, v in inputs.items():
        output_data[k] = v.to(device)

    return output_data


def truncate_sentences(
    sentence_list: List[str], maximum_length: Union[int, float]
) -> List[str]:
    """Truncates list of sentences to a maximum length.

    Args:
        sentence_list (List[str]): List of sentences to be truncated.
        maximum_length (Union[int, float]): Maximum length of any output sentence.

    Returns:
        List[str]: List of truncated sentences.
    """

    truncated_sentences = []

    for sentence in sentence_list:
        truncated_sentences.append(sentence[:maximum_length])

    return truncated_sentences


def pool_and_normalize(
    features_sequence: torch.Tensor,
    attention_masks: torch.Tensor,
    return_norms: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Temporal ooling of sequences of vectors and projection onto the unit sphere.

    Args:
        features_sequence (torch.Tensor): Inpute features with shape [B, T, F].
        attention_masks (torch.Tensor): Pooling masks with shape [B, T, F].
        return_norms (bool, optional): Whether to additionally return the norms. Defaults to False.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: Pooled and normalized vectors with shape [B, F].
    """

    pooled_embeddings = pooling(features_sequence, attention_masks)
    embedding_norms = pooled_embeddings.norm(dim=1)

    normalizing_factor = torch.where(  # Only normalize embeddings with norm > 1.0.
        embedding_norms > 1.0, embedding_norms, torch.ones_like(embedding_norms)
    )

    pooled_normalized_embeddings = pooled_embeddings / normalizing_factor[:, None]

    if return_norms:
        return pooled_normalized_embeddings, embedding_norms
    else:
        return pooled_normalized_embeddings


def pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pools a batch of vector sequences into a batch of vector global representations.
    It does so by taking the last vector in the sequence, as indicated by the mask.

    Args:
        x (torch.Tensor): Batch of vector sequences with shape [B, T, F].
        mask (torch.Tensor): Batch of masks with shape [B, T].

    Returns:
        torch.Tensor: Pooled version of the input batch with shape [B, F].
    """

    eos_idx = mask.sum(1) - 1
    batch_idx = torch.arange(len(eos_idx), device=x.device)

    mu = x[batch_idx, eos_idx, :]

    return mu


def prepare_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=True)

    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.add_special_tokens({"sep_token": SEPARATOR_TOKEN})
    tokenizer.add_special_tokens({"cls_token": CLS_TOKEN})
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer


class StarEncoder(torch.nn.Module, ABC):
    def __init__(self, device, max_input_len, maximum_token_len):
        super().__init__()

        self.model_name = "bigcode/starencoder"
        self.tokenizer = prepare_tokenizer(self.model_name)
        self.encoder = (
            AutoModel.from_pretrained(self.model_name, token=True)
            .to(DEVICE)
            .eval()
        )
        self.device = device
        self.max_input_len = max_input_len
        self.maximum_token_len = maximum_token_len

    def forward(self, input_sentences):
        inputs = self.tokenizer(
            [f"{CLS_TOKEN}{sentence}{SEPARATOR_TOKEN}" for sentence in input_sentences],
            padding="longest",
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.encoder(**set_device(inputs, self.device))
        embedding = pool_and_normalize(outputs.hidden_states[-1], inputs.attention_mask)

        return embedding

    def encode(self, input_sentences, batch_size=32, **kwargs):
        truncated_input_sentences = truncate_sentences(
            input_sentences, self.max_input_len
        )

        n_batches = len(truncated_input_sentences) // batch_size + int(
            len(truncated_input_sentences) % batch_size > 0
        )

        embedding_batch_list = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(truncated_input_sentences))

            with torch.no_grad():
                embedding_batch_list.append(
                    self.forward(truncated_input_sentences[start_idx:end_idx])
                    .detach()
                    .cpu()
                )

        input_sentences_embedding = torch.cat(embedding_batch_list)

        return [emb.squeeze().numpy() for emb in input_sentences_embedding]


if __name__ == "__main__":
    data_dir = "../data/"
    polybench_dir = "../../polybench/"

    properties_file = os.path.join(data_dir, "gcc/others/properties.csv")
    new_properties_file = properties_file.replace(
        "properties.csv", "input_embeddings.csv"
    )

    encoder = StarEncoder(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)

    embeddings = []

    for s in tqdm(pd.read_csv(properties_file).name):
        paths = glob.glob(os.path.join(polybench_dir, f"**/{s}.c"), recursive=True)
        assert len(paths) == 1, f"Found multiple results for {s}: {paths}"
        f = open(paths[0]).read()
        enc = encoder.encode([f])
        embeddings.append([s] + enc[0].tolist())

    df = pd.DataFrame(
        embeddings, columns=["name"] + [f"v{i}" for i in range(enc[0].shape[-1])]
    )
    df.to_csv(new_properties_file, index_label="id")
