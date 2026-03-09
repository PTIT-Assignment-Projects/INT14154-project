import pandas as pd
from transformers import AutoTokenizer

class TextProcessor:
    def __init__(self, model_name="distilbert-base-uncased", max_len=128):
        """
        Initializes the TextProcessor with a pretrained transformer tokenizer.

        Args:
            model_name (str): The name of the pretrained model/tokenizer.
            max_len (int): Maximum length for tokenization.
        """
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentence_model = None

    def tokenize(self, texts, return_tensors="pt"):
        """
        Tokenizes a list of texts into input_ids and attention_mask.

        Args:
            texts (list, str or pd.Series): The text or list of texts to tokenize.
            return_tensors (str): The type of tensors to return ('pt' for PyTorch).

        Returns:
            dict: A dictionary containing input_ids and attention_mask.
        """
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            # Convert series to list and fill NaN values with empty string
            texts = texts.fillna("").tolist()
        elif isinstance(texts, list):
            # Ensure all elements are strings and handle potential None/NaN in list
            texts = [str(t) if t is not None else "" for t in texts]

        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors=return_tensors
        )