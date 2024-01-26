import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import codecs
from peft import PeftModel, PeftConfig
import sys
import json
import os
path = sys.argv[1]
save_path = sys.argv[2]
with open(path,'r') as f:
      data = f.readlines()

data = [entry for entry in data if entry != '\n']
import transformers
from transformers import AutoTokenizer
from flan_ext import T5ForExtractiveSummarization
import torch
model_name = "/home/atr1n17/.cache/huggingface/hub/google__flan-t5-base.7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
adapter_name = "/scratch/atr1n17/COLING2022/models/idn_corrected_tokens/flan_ext_layernorm/checkpoint-2000"
model = T5ForExtractiveSummarization.from_pretrained(model_name, max_sent_num=16)
adapter_config = PeftConfig.from_pretrained(adapter_name)
model = PeftModel.from_pretrained(model, adapter_name)
model.cuda()
#MAX_SENT_LEN = 16
with codecs.open(os.path.join(save_path, 'hyp.txt'),'w', encoding='utf-8', errors='ignore') as f:
   pass

#summaries = []
for i,entry in enumerate(data):
  #print(i)
  text = json.loads(entry)['doc'] 
  summary = ''
  
  sentences = sent_tokenize(text)
  all_sents = []
  all_probs = []
  for index in range(0, len(sentences),25):
        
        options = sentences[index:index+25]
        
        #options = "Create an extractive summary for the following. Document: " + options + " Extractive Summary: "
        tokenized_input = tokenizer(options,max_length=16,padding='max_length', return_tensors="pt", truncation=True)
        input_ids = tokenized_input.input_ids
        attention_mask = tokenized_input.attention_mask
        #tokenizer(options, max_length=384, padding='max_length', truncation=True, return_tensors="pt").input_ids
        
       
        #input_ids = tokenizer.encode(f"{INSTRUCTION}\n{PROMPT}\n Extractive Summary:", return_tensors="pt", truncation=True)
        #print(input_ids.shape)
        #input_ids = input_ids.to('cuda')
        
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        output = model(input_ids=input_ids, attention_mask = attention_mask)['logits']
        probs = torch.sigmoid(output).squeeze()
        # Check if probs is a single float value or a tensor
        if probs.dim() == 0:  # single value
            probs = [probs.item()]
        else:  # tensor
            probs = probs.detach().cpu().tolist()

        # Extend the all_sentences and all_probs lists

        all_sents.extend(options)
        all_probs.extend(probs)

        # Clear CUDA cache
        torch.cuda.empty_cache()

  # Select overall top-k sentences
  top_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)[:81]

  # Sort top_indices to maintain the original order of sentences
  top_indices.sort()

  # Retrieve the top sentences in their original order
  top_sentences = [all_sents[i] for i in top_indices]
  summary = ' '.join(top_sentences)  # Select overall top-k sentences
  #sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)[:81]
  #top_sentences = [all_sents[i] for i in sorted_indices]

 # summary = ' '.join(top_sentences)

  with codecs.open(os.path.join(save_path, 'hyp.txt'), 'a', encoding='utf-8', errors='ignore') as f:
        f.write(summary + '<<END>>')
        #probs = torch.sigmoid(output)
        #probs = probs.float()
        #all_probs.append(probs)
  
  #k_largest = heapq.nlargest(81, all_probs)
  #indices = [all_probs.index(x) for x in k_largest]
  #indices.sort()
  #summary_sentences = [sentence for sentence in sentences if sentences.index(sentence) in indices]
  #summary = ' '.join(summary_sentences)
  #with codecs.open(os.path.join(save_path, 'hyp.txt'),'a', encoding='utf-8', errors='ignore') as f:
  #    f.write(summary+'<<END>>')
  #summaries.append(summary)
#with codecs.open(os.path.join(save_path, 'hyp.txt'),'w', encoding='utf-8', errors='ignore') as f:
#      f.write('<<END>>'.join(summaries))
  


