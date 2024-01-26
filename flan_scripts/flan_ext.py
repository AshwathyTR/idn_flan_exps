from datasets import Dataset, DatasetDict
import json
import sys
#save_dir = sys.argv[1]

def load_dataset(file_path, chunk_size=25):
    # Read the data from the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    #print(len(lines))
    # Initialize lists to store the data
    docs = []
    targets = []
    rationale_targets = []

    # Process each line in the file
    for line in lines:
        data = json.loads(line)
        sentences = data['doc'].split('\n')  # Assuming sentences end with '. '
        labels = data['labels'].split('\n')  # Assuming labels are also split by sentences
        rationales = data['rationale_w'].split('\n')
    
        # Split sentences and labels into chunks of 'chunk_size'
        sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        label_chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)]
        rationale_chunks = [rationales[i:i + chunk_size] for i in range(0, len(rationales), chunk_size)]
        # Process each chunk
        for chunk, label_chunk, rationale_chunk in zip(sentence_chunks, label_chunks,rationale_chunks):
            chunk_doc = '\n'.join(chunk)
            chunk_target = '\n'.join(label_chunk)
            chunk_rationale = '\n'.join(rationale_chunk)
            if '1' not in  chunk_target:
                 continue
            docs.append(chunk_doc)
            targets.append(chunk_target)
            rationale_targets.append(chunk_rationale)
    
    print(len(docs))
    # Create a Dataset object
    dataset = Dataset.from_dict({
        'source': docs,
        'target': targets,
        'rationale': rationale_targets
    })

    # Return the dataset
    return dataset

# Usage example for training data
train_file_path = "/scratch/atr1n17/COLING2022/data/idn_corrected_tokens/sr_81/train_ws_2_20.json"
eval_file_path = "/scratch/atr1n17/COLING2022/data/idn_corrected_tokens/sr_81/val_ws_2_20.json"
MAX_SENT_LEN = 16
train_dataset = load_dataset(train_file_path)
eval_dataset = load_dataset(eval_file_path)
# Wrap the train dataset in a DatasetDict if needed
dataset = DatasetDict({'train': train_dataset, "validation": eval_dataset})

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
model_name = "/home/atr1n17/.cache/huggingface/hub/google__flan-t5-base.7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# data preprocessing
#text_column = "source"
#label_column = "target"
#MAX_SENT_LEN = 25
import torch
import torch.nn.functional as F

# Function to tokenize the documents and prepare labels
def preprocess_function(examples):
    tokenized_examples = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
        
    }

    for i, doc in enumerate(examples['source']):
        # Split document into sentences
        sentences = doc.split('\n')
        # Tokenize sentences
        tokenized_sentences = tokenizer(sentences, max_length=MAX_SENT_LEN, padding='max_length', truncation=True, return_tensors="pt")
        #print(tokenized_sentences.input_ids.shape) #[num_sentences, sent_len]
        # Add tokenized sentences to our example
        tokenized_examples['input_ids'].append(tokenized_sentences.input_ids.flatten())
        tokenized_examples['attention_mask'].append(tokenized_sentences.attention_mask.flatten())
   
        labels = examples['target'][i].split('\n')
        labels = [int(label) for label in labels]
        tokenized_examples['labels'].append(labels)
   

    return tokenized_examples


# Apply the preprocessing function to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['source', 'target'])

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

import torch
import torch.nn as nn
from transformers import T5EncoderModel
class T5ForExtractiveSummarization(T5EncoderModel):
    def __init__(self, config, max_sent_num):
        super().__init__(config)
        self.max_sent_num = max_sent_num
        self.classifier = nn.Linear(config.d_model, 1)
        self.sent_attention = nn.Linear(config.d_model, 1)  # Attention mechanism for sentences
        #self.batch_norm = nn.BatchNorm1d(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, rationales=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Handle inputs_embeds
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Encoder outputs
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs.last_hidden_state
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Reshape to [batch_size, num_sentences, MAX_SENT_LEN, hidden_size]
        num_sentences = seq_length // self.max_sent_num
        hidden_states = hidden_states.view(batch_size, num_sentences, self.max_sent_num, hidden_size)

        # Apply attention pooling
        sent_attention_weights = torch.softmax(self.sent_attention(hidden_states), dim=2)  # [batch_size, num_sentences, MAX_SENT_LEN, 1]
        sent_attention_weights = sent_attention_weights * attention_mask.view(batch_size, num_sentences, self.max_sent_num, 1)  # Mask padding tokens
        sentence_representations = torch.sum(sent_attention_weights * hidden_states, dim=2)  # [batch_size, num_sentences, hidden_size]
        #print(sentence_representations.shape)
        #sentence_representations = self.batch_norm(sentence_representations.transpose(1, 2)).transpose(1, 2)
        sentence_representations = self.layer_norm(sentence_representations)
        #print(sentence_representations.shape)
        # Classifier
        logits = self.classifier(sentence_representations).squeeze(-1)  # [batch_size, num_sentences]
        #print(logits)
        # Loss computation
        if labels is not None:
            labels = labels.to(torch.float)
            #print(labels)
            #print(logits)
            # Create a mask for non-padding labels
            logits = logits.view(-1)
            labels = labels.view(-1)
            mask = (labels != -100).to(torch.float)

            # Apply the mask
            filtered_logits = logits[mask == 1]
            filtered_labels = labels[mask == 1]

            # Loss function
            pos_weight = torch.tensor([2.0])
            pos_weight = pos_weight.to('cuda')
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
            #loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(filtered_logits, filtered_labels)
            #loss_fct = torch.nn.BCEWithLogitsLoss()
            #loss = loss_fct(logits.view(-1), labels.view(-1))
            #slurm-5023952.outprint(loss)
            return {"loss": loss, "logits": logits, "attn":sent_attention_weights}
        
        #print("no labels")
        return {"logits": logits, "attn":sent_attention_weights}



# Select CUDA device index
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

#model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#config = T5EncoderModel.from_pretrained(model_name)

model = T5ForExtractiveSummarization.from_pretrained(model_name, max_sent_num=MAX_SENT_LEN)
#model.config.dropout_rate = 0.1
from peft import LoraConfig, get_peft_model, TaskType


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], modules_to_save=['sent_attention','classifier','layer_norm'], lora_dropout=0.05, bias="none", task_type="SEQ_CLS"
)


model = get_peft_model(model, lora_config)
model.cuda()
print_trainable_parameters(model)
from torch.nn.utils.rnn import pad_sequence
import torch
from math import ceil

def custom_loss_function(outputs, labels):
    # Standard classification loss
    labels = labels.to(torch.float)
    pos_weight = torch.tensor([2.0]) 
    pos_weight = pos_weight.to('cuda')
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    logits = outputs["logits"].view(-1)
    labels = labels.view(-1)
    mask = (labels != -100).to(torch.float)

            # Apply the mask
    filtered_logits = logits[mask == 1]
    filtered_labels = labels[mask == 1]

    loss = loss_fct(outputs["logits"].view(-1), labels.view(-1))

    # Custom loss between attention weights and rationales
    # Assuming rationales are provided in a similar shape as outputs["attn"]
    #attention_loss_fct = torch.nn.BCEWithLogitsLoss()
    #attention_loss = attention_loss_fct(outputs["attn"], rationales)

    # Combine the losses
    #combined_loss = classification_loss + attention_loss
    return loss

def custom_collate_fn(features, max_len=512):
    # Determine the length to which the labels should be padded/truncated
    label_max_len = ceil(max_len / MAX_SENT_LEN)

    # Pad or truncate input_ids and attention_mask
    input_ids = pad_sequence([torch.tensor(f["input_ids"][:max_len], dtype=torch.long) for f in features],
                             batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(f["attention_mask"][:max_len], dtype=torch.long) for f in features],
                                  batch_first=True, padding_value=0)

    # Pad or truncate labels
    labels = pad_sequence([torch.tensor(f["labels"][:label_max_len], dtype=torch.float) for f in features],
                          batch_first=True, padding_value=-100)  # -100 is often used for ignored index in PyTorch

    # Adjust the attention mask for additional padding tokens
    # If you're adding padding tokens, you should ensure the attention mask is 0 for those tokens
    attention_mask = attention_mask.bool() & input_ids.ne(tokenizer.pad_token_id)
    #rationales = pad_sequence([torch.tensor(f["rationales"][:label_max_len], dtype=torch.float) for f in features],
    #                         batch_first=True, padding_value=tokenizer.pad_token_id)
    #rationales = rationales.unsqueeze(-1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
from transformers import TrainingArguments, Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        labels = inputs.get("labels")
        #rationales = inputs.get("rationales")  # Assuming rationales are part of your dataset
        loss = custom_loss_function(outputs, labels)
        return (loss, outputs) if return_outputs else loss

#training_args = TrainingArguments(
    #output_dir="/scratch/atr1n17/COLING2022/models/idn_corrected_tokens/flan_ext_layernorm",
#    output_dir = save_dir,
#    evaluation_strategy="steps",
#    learning_rate=1e-4,
#    gradient_accumulation_steps=16,
#    per_device_train_batch_size=4,
#    remove_unused_columns=False,
#    num_train_epochs=3,
#    save_steps=500,
#    eval_steps=500,
#    save_strategy = "steps",
#    warmup_steps = 10000,
#    weight_decay = 0.01
#)

# Use the custom trainer for training
#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=train_dataset,
#    eval_dataset=eval_dataset,
    #compute_loss = custom_loss_function,
#    data_collator = custom_collate_fn,
#)
if __name__ == "__main__":
  save_dir = sys.argv[1]
  training_args = TrainingArguments(
    #output_dir="/scratch/atr1n17/COLING2022/models/idn_corrected_tokens/flan_ext_layernorm",
    output_dir = save_dir,
    evaluation_strategy="steps",
    learning_rate=1e-4,
    gradient_accumulation_steps=16,
    per_device_train_batch_size=4,
    remove_unused_columns=False,
    num_train_epochs=3,
    save_steps=500,
    eval_steps=500,
    save_strategy = "steps",
    warmup_steps = 10000,
    weight_decay = 0.01
    )   
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #compute_loss = custom_loss_function,
    data_collator = custom_collate_fn,
  )
  trainer.train()


