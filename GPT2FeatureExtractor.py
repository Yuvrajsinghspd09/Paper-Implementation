# STEP_1  - Setting up the Env
pip install torch transformer pyyaml

import torch
import torch.nn as nn
from transformers load GPT2Toeknizer, GPT2Model
from torch.utils.data import DataLoader, TensorDataset


# STEP_2 GPT2FeatureExtractor

'''
 Implement GPT2FeatureExtractor:
   * Use transformers library to load GPT-2 model and tokenizer
   * Implement methods to extract and process hidden states
'''


class GPT2FeatureExtractor:
  def __init__(self,model_name='gpt2'):
    self.tokenizer=GPT2Tokenizer.from_pretrained(model_name)
    self.model= GPT2Model.from_pretrained(model_name)

  def extract_features(self,text):
    inputs = self.tokenizer(text,return_tensors='pt',padding=True,truncation=True)
    with torch.no_grad():
      output= self.model(**inputs)
      return outputs.last_hidden_state.mean(dim=1)



