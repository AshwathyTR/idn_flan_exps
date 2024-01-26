

model_name = "/home/atr1n17/.cache/huggingface/hub/google__flan-t5-base.7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"

#from datasets import Dataset, DatasetDict
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, AutoModel
import torch
print(torch.version.cuda)

#model_name = "google/flan-t5-base"
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,  config=config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
)


model = get_peft_model(model, lora_config)
model.cuda()
print_trainable_parameters(model)


from datasets import Dataset, DatasetDict
import json
def convert_to_dataset_dict(source_file, dataset_name="My Dataset", chunk_size=25):
    with open(source_file, 'r') as f:
        h = f.readlines()

    #print(len(h))
    docs = []
    targets = []
    empty_targets = 0
    for line in h:
        data = json.loads(line)
        sentences = data['doc'].split('\n')  # Assuming sentences end with '. '
        labels = data['labels'].split('\n')  # Assuming labels are also split by sentences

        # Split sentences and labels into chunks of 'chunk_size'
        sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
        label_chunks = [labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)]

        # Process each chunk
        for chunk, label_chunk in zip(sentence_chunks, label_chunks):
            chunk_doc = '. '.join(chunk)
            chunk_target = '. '.join([chunk[i] for i, l in enumerate(label_chunk) if l == '1'])
            if not chunk_target.strip():
               continue
            chunk_doc = "Create an extractive summary for the following. Document: " + chunk_doc + " Extractive Summary: "
            #docs.append(chunk_doc)
            #targets.append(chunk_target)
            #print(chunk_target)
            #if not chunk_target:
            #   empty_targets = empty_targets + 1
            #if not chunk_target:
            #   continue
            docs.append(chunk_doc)
            targets.append(chunk_target)


    print("EMPTY: "+ str(empty_targets))
    print("TOTAL: "+ str(len(targets)))# Create a Dataset object
    dataset = Dataset.from_dict({
        'source': docs,
        'target': targets
    })

    # Create and return a DatasetDict object
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict




# Example usage
source_file = "/scratch/atr1n17/COLING2022/data/idn_corrected_tokens/sr_81/train.json"
train_dataset = convert_to_dataset_dict(source_file)
source_file = "/scratch/atr1n17/COLING2022/data/idn_corrected_tokens/sr_81/val500.json"
eval_dataset = convert_to_dataset_dict(source_file)

# data preprocessing
text_column = "source"
label_column = "target"
max_length = 384
import torch.nn.functional as F
#model_name = "google/flan-t5-base"
#tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]

    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
    decoder_inputs_and_labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True)

    # Convert input_ids to lists and handle labels
    labels = decoder_inputs_and_labels["input_ids"].copy()
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]

    # Update model_inputs with labels and decoder input IDs converted to lists
    model_inputs["labels"] = labels
    model_inputs["decoder_input_ids"] = decoder_inputs_and_labels["input_ids"]





    return model_inputs

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    )



train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=train_dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)["train"]
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=eval_dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)["train"]


from torch.nn.utils.rnn import pad_sequence
import torch
import transformers
#from rouge_score import rouge_scorer

from transformers import TrainingArguments, Trainer
'''
def compute_metrics(eval_pred):
    #return 0
    predictions, labels = eval_pred
    print(len(predictions))
    print(predictions[0].shape)
    # Decode predictions and labels from their token IDs if necessary
    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True) for p in predictions]
    decoded_labels = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True) for l in labels]

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    rouge_results = [] 
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(pred, label)
        rouge_results.append( scores['rouge1'].fmeasure)
        #rouge_results['rouge2'].append(scores['rouge2'].fmeasure)
        #rouge_results['rougeL'].append(scores['rougeL'].fmeasure)

    # Calculate mean ROUGE scores
    mean_rouge_results = np.mean(rouge_results) 
    return mean_rouge_results
'''
training_args = TrainingArguments(
    output_dir="/scratch/atr1n17/COLING2022/models/idn_corrected_tokens/flan_nomt",
    evaluation_strategy="steps",
    learning_rate=1e-4,
    gradient_accumulation_steps=16,
    per_device_train_batch_size=4,

    remove_unused_columns=False,
    num_train_epochs=3,
    save_steps=500,
    eval_steps = 500,
    save_strategy = "steps",
    warmup_steps = 10000,
    weight_decay = 0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #compute_metrics=compute_metrics
)
trainer.train()


























