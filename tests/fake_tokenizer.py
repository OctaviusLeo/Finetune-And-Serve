"""A small chat tokenizer stand-in for CPU tests (no download): characters map to ids 3..102."""

from __future__ import annotations

import torch


class FakeChatTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self):
        self.padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return {"input_ids": self._ids(text)}
        return text

    @staticmethod
    def _ids(text):
        return [3 + (ord(c) % 100) for c in text]

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": self._ids(texts)}
        ids = [self._ids(t) for t in texts]
        width = max(len(x) for x in ids)
        left = self.padding_side == "left"
        padded = [([0] * (width - len(x)) + x) if left else (x + [0] * (width - len(x))) for x in ids]
        mask = [([0] * (width - len(x)) + [1] * len(x)) if left else ([1] * len(x) + [0] * (width - len(x))) for x in ids]
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(mask)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)
