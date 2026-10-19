from __future__ import annotations

import copy

import torch
import torch.nn as nn


class SoftVerbalizer(nn.Module):
    """OpenPrompt 1.0.1 ``SoftVerbalizer`` subset used by DPT.

    DPT only asks the verbalizer to copy the masked-language-model head,
    replace its vocabulary decoder with a trainable per-level decoder, and
    project hidden states through that head.  Keeping this small compatible
    implementation avoids importing the unrelated OpenPrompt generation stack,
    whose Transformers 4.11-era imports no longer exist in current releases.
    """

    def __init__(self, tokenizer, model: nn.Module, classes: list[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.classes = classes
        self.num_classes = len(classes)

        children = list(model.named_children())
        if not children:
            raise ValueError("The masked-language model must expose a prediction head")
        head_name, head = children[-1]
        self.head = copy.deepcopy(head)

        module = self.head
        parent: nn.Module | None = None
        last_name: str | None = None
        last_linear: nn.Linear | None = None
        for _ in range(5):
            named_children = list(module.named_children())
            if not named_children:
                break
            last_name, child = named_children[-1]
            parent = module
            module = child
            if isinstance(child, nn.Linear):
                last_linear = child
                break
        if parent is None or last_name is None or last_linear is None:
            raise RuntimeError(f"Cannot locate the decoder in model head {head_name!r}")

        # Match OpenPrompt 1.0.1 exactly: the per-class decoder is bias-free.
        replacement = nn.Linear(last_linear.in_features, self.num_classes, bias=False)
        setattr(parent, last_name, replacement)
        # BertLMPredictionHead retains a vocabulary-sized ``bias`` parameter
        # after OpenPrompt replaces ``decoder``.  It is unused because the new
        # decoder is bias-free, so leave it untouched for state-dict parity.
        if hasattr(parent, "bias"):
            parent.decoder.bias = None

    def process_outputs(self, outputs: torch.Tensor, batch=None, **kwargs) -> torch.Tensor:
        return self.head(outputs)
