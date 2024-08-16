import os

from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

train_texts = [
    "I love machine learning.",
    "Transformers are amazing!",
    "I don't like the new update.",
    "Natural language processing is fascinating."
]

train_labels = [1, 1, 0, 1]  # 假设1代表正面情感，0代表负面情感


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_dataset = CustomDataset(train_encodings, train_labels)

current_file_path = os.path.abspath(__file__)

current_file_name = os.path.basename(current_file_path)

training_args = TrainingArguments(
    output_dir='./results'+current_file_name,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs'+current_file_name,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
