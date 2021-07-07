from transformers import AutoModel
import torch.nn as nn
from torch.cuda.amp import autocast

class SentencePairClassifier(nn.Module):
    """SentnecePairClassifier takes two sentences seperately and concatenates them with special tokens. 
    Then it feeds concateneated sentence to a chosen Bert model and a classification layer."""
    
    def __init__(self, bert_model='albert-base-v2', freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
     
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "bert-base-uncased": # 110M parameters
            hidden_size = 768
        elif bert_model == 'distilbert-base-uncased':
            hidden_size = 768
        elif bert_model == 'google/electra-small-discriminator':
            hidden_size = 256

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        _, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)
        #pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits