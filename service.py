from transformers import AutoTokenizer
import torch
from model import SentencePairClassifier
#tokenizer = AutoTokenizer.from_pretrained(bert_model)
"""
parent_ex = input("Enter Parent sentece: ")
text_ex = input("Enter Target sentence: ")
bert_model = input("Enter model name: (i.e 'distilbert-base-uncased')")
tokenizer = AutoTokenizer.from_pretrained(bert_model)
maxlen = 500
encoded_texts = tokenizer.encode_plus(parent_ex, text_ex, max_length=maxlen,
                                     add_special_tokens=True,
                                     return_token_type_ids=True,
                                     pad_to_max_length=True,
                                     return_attention_mask=True,
                                     truncation=True,
                                     return_tensors='pt')


input_ids = encoded_texts['input_ids']
attention_mask = encoded_texts['attention_mask']
token_type_ids = encoded_texts['token_type_ids']

path_to_model = '/content/models/distilbert-base-uncased_lr_2e-05_val_loss_0.06106_ep_4.pt'  
device2 = torch.device('cpu')
model = SentencePairClassifier(bert_model)
model.load_state_dict(torch.load(path_to_model, map_location=device2))
output = model(input_ids, attention_mask, token_type_ids)
_, prediction = torch.max(output, dim=1)

class_name = ["Non-toxic","Toxic"]

print("Parent sentence: ", parent_ex)
print("Target sentence: ", text_ex)
print("Model prediction:", class_name[prediction])

"""
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
        tokenizer = AutoTokenizer.from_pretrained(parsed_json['model_name'])
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
        path_to_model = './models/distilbert-base-uncased_lr_2e-05_val_loss_0.06106_ep_4.pt'  
        
        device2 = torch.device('cpu')
        model = self.artifacts.model
        model.load_state_dict(torch.load(path_to_model, map_location=device2)) #이렇게 self.artifact 붙이면 되나?
        output = model(input_ids, attention_mask, token_type_ids)
        _, prediction = torch.max(output, dim=1)

        class_name = ["Non-toxic","Toxic"]
        
        #results = self.artifacts.model.predict(parsed_json) ##여기를 바꿔야함
        return class_name[prediction]
