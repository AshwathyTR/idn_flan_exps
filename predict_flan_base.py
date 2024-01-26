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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
import torch
model_name = "/home/atr1n17/.cache/huggingface/hub/google__flan-t5-base.7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#adapter_name = "/scratch/atr1n17/COLING2022/models/idn_corrected_tokens/flan_nomt/checkpoint-3500"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#adapter_config = PeftConfig.from_pretrained(adapter_name)
#model = PeftModel.from_pretrained(model, adapter_name)
model.cuda()
with codecs.open(os.path.join(save_path, 'hyp.txt'),'w', encoding='utf-8', errors='ignore') as f:
   pass

#summaries = []
for i,entry in enumerate(data):
  #if i<3347:
  #   continue
  #print(i)
  text = json.loads(entry)['doc'] 
  summary = ''
  
  sentences = sent_tokenize(text)
  for index in range(0, len(sentences),25):
        
        options = '. '.join(sentences[index:index+25])
        
        options = "Create an extractive summary for the following. Document: " + options + " Extractive Summary: "
        input_ids = tokenizer.encode(options,max_length=384, return_tensors="pt", truncation=True)
        #tokenizer(options, max_length=384, padding='max_length', truncation=True, return_tensors="pt").input_ids
        
       
        #input_ids = tokenizer.encode(f"{INSTRUCTION}\n{PROMPT}\n Extractive Summary:", return_tensors="pt", truncation=True)
        #print(input_ids.shape)
        #input_ids = input_ids.to('cuda')
        
        input_ids = input_ids.to('cuda')
        output = model.generate(input_ids=input_ids, max_new_tokens=128,repetition_penalty=2.0)
        #output_text = tokenizer.decode(output[0], skip_special_tokens=True).split('Extractive Summary:')[1]
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        #print(output_text)
        summary = summary +' '+ output_text
  with codecs.open(os.path.join(save_path, 'hyp.txt'),'a', encoding='utf-8', errors='ignore') as f:
      f.write(summary+'<<END>>')
  #summaries.append(summary)
#with codecs.open(os.path.join(save_path, 'hyp.txt'),'w', encoding='utf-8', errors='ignore') as f:
#      f.write('<<END>>'.join(summaries))
  


