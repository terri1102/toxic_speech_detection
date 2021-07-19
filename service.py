from transformers import AutoTokenizer
import torch
from model import SentencePairClassifier
from transformers import DistilBertTokenizerFast
#tokenizer = AutoTokenizer.from_pretrained(bert_model)

import pandas as pd
import bentoml
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from typing import List
from bentoml.types import JsonSerializable
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import FileInput, JsonOutput

class_name = ["Non-toxic","Toxic"]

@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact('model')])
class ToxicspeechClassifier(BentoService):

    @api(input=JsonInput()) #{'parent': "This hate comment nasty.", 'text': "also toxic", 'model_name': "distilbert-base-uncased"}
    def predict(self, parsed_json):
        parent = parsed_json['parent']
        text = parsed_json['text']
        
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased') #albert-base-v2, distilbert-base-uncased
        encoded_texts = tokenizer.encode_plus(parent, text, max_length=500,
                                     add_special_tokens=True,
                                     return_token_type_ids=True,
                                     pad_to_max_length=True,
                                     return_attention_mask=True,
                                     truncation=True,
                                     return_tensors='pt')
        input_ids = encoded_texts['input_ids']
        attention_mask = encoded_texts['attention_mask']
        token_type_ids = encoded_texts['token_type_ids']
        

        #parallel 로 훈련한 모델의 module 제거
      #  from collections import OrderedDict
      # new_state_dict = OrderedDict()
      #  for k, v in state_dict.items():
      #      name = k[7:] # remove `module.`
      #      new_state_dict[name] = v

        # load params
      #  model = self.artifacts.model
       # model.load_state_dict(state_dict, strict=False)
        
      #  output = model(input_ids, attention_mask, token_type_ids)
        model_output = self.artifacts.model(input_ids, attention_mask, token_type_ids)
        if model_output < 0:
            answer = "Non-Toxic"
        else:
            answer = "Toxic"
    
        return answer
