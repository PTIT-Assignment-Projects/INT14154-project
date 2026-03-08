from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constant import LABEL_COLUMNS, TEXT_COLUMN
from src.preprocessing import TextProcessor


class ToxicDataset(Dataset):
    def __init__(self, df, processor, text_column=TEXT_COLUMN, label_columns=None):
        """
        PyTorch Dataset for toxic comment classification.

        Args:
            df (pd.DataFrame): The dataframe containing texts and labels.
            processor (TextProcessor): Instance of TextProcessor.
            text_column (str): Name of the text column.
            label_columns (list): List of label column names.
        """
        self.df: pd.DataFrame = df
        self.processor: TextProcessor = processor
        self.text_column: Optional[str] = text_column
        self.label_columns: Optional[list] = label_columns or LABEL_COLUMNS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Handle indexing correctly for both series and dataframes
        row = self.df.iloc[idx]
        text = str(row[self.text_column]) if pd.notnull(row[self.text_column]) else ""

        # Check if labels exist in the dataframe (for test sets without labels)
        if all(col in self.df.columns for col in self.label_columns):
            labels = torch.tensor(row[self.label_columns].values.astype(float), dtype=torch.float)
        else:
            labels = torch.tensor([])

        encoding = self.processor.tokenize(text, return_tensors="pt")

        # Flatten the tensors since tokenize returns (batch_size=1, max_len) for single string
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }