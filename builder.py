from service import ToxicspeechClassifier
from model import SentencePairClassifier
import torch
#from transformers import DistilBertConfig

path_to_model = './models/distilbert-base-uncased_lr_2e-05_val_loss_0.03533_ep_2.pt'
#def saveToBento(checkpoint):
    #model_state_dict, _, _, _, _, = utils.load_model(checkpoint)

#define the model
bert_model = SentencePairClassifier(bert_model='distilbert-base-uncased') #'albert-base-v2' 'distilbert-base-uncased'

#load saved model
bert_model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')), strict=False) #, map_location=device2
#bert_model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))

#Add model artifact to BentoML
bento_svc = ToxicspeechClassifier()
bento_svc.pack('model', bert_model)

#Save Bento Service
saved_path = bento_svc.save()
print('Bento Service saved in', saved_path)
