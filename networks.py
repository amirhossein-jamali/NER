import torch
import torch.nn as nn
from transformers import AutoConfig
from utils import match_kwargs


class NERNetwork(nn.Module):
    """A Generic Network for NER models.

    Can be replaced with a custom user-defined network with
    the restriction, that it must take the same arguments.
    """

    def __init__(self, transformer: nn.Module, device: str, n_tags: int, dropout: float = 0.1) -> None:
        """Initialize a NER Network

        Args:
            transformer (nn.Module): huggingface `torch` transformer.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(NERNetwork, self).__init__()

        # extract transformer name
        transformer_name = transformer.name_or_path
        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(transformer_name)

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.tags = nn.Linear(transformer_config.hidden_size, n_tags)
        self.device = device

    # NOTE: 'offsets 'are not used in model as-is, but they are expected as output
    # down-stream. So _DON'T_ remove! :)
    def forward(self,
                input_ids: torch.Tensor,
                masks: torch.Tensor,
                token_type_ids: torch.Tensor,
                target_tags: torch.Tensor,
                offsets: torch.Tensor) -> torch.Tensor:
        """Model Forward Iteration

        Args:
            input_ids (torch.Tensor): Input IDs.
            masks (torch.Tensor): Attention Masks.
            token_type_ids (torch.Tensor): Token Type IDs.
            target_tags (torch.Tensor): Target tags. Are not used
                in model as-is, but they are expected downstream,
                so they can not be left out.
            offsets (torch.Tensor): Offsets to keep track of original
                words. Are not used in model as-is, but they are
                expected as down-stream, so they can not be left out.

        Returns:
            torch.Tensor: predicted values.
        """

        transformer_inputs = {
            'input_ids': input_ids.to(self.device),
            'masks': masks.to(self.device),
            'token_type_ids': token_type_ids.to(self.device)
        }

        # match args with transformer
        transformer_inputs = match_kwargs(self.transformer.forward, **transformer_inputs)

        outputs = self.transformer(**transformer_inputs)[0]

        # apply drop-out
        outputs = self.dropout(outputs)

        # outputs for all labels/tags
        outputs = self.tags(outputs)

        return outputs
