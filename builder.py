from service import ToxicspeechClassifier
from model import SentencePairClassifier
import torch
from transformers import DistilBertConfig

path_to_model = './models/albert-base-v2_lr_2e-05_val_loss_0.03766_ep_2.pt'
#def saveToBento(checkpoint):
    #model_state_dict, _, _, _, _, = utils.load_model(checkpoint)

#define the model
bert_model = SentencePairClassifier(bert_model='albert-base-v2') #'albert-base-v2' 'distilbert-base-uncased'

#load saved model
bert_model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')), strict=False) #, map_location=device2

#Add model to BentoML
bento_svc = ToxicspeechClassifier()
bento_svc.pack('model', bert_model)

#Save Bento Service
saved_path = bento_svc.save()
print('Bento Service saved in', saved_path)


#if __name__ == "__main__":

    #classifier service instance 생성
 #   classifier_service = ToxicspeechClassifier()

    #model artifact로 pack
  #  classifier_service.pack("model", SentencePairClassifier)

    #모델 서빙을 위해 prediction 저장
   # saved_path = classifier_service.save()