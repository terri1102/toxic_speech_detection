from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AlbertTokenizerFast
from torch.utils.data import DataLoader, Dataset
import argparse

class CustomDataset(Dataset):
    
    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):

        self.data = data  
        
        #self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  
        self.tokenizer = AlbertTokenizerFast.from_pretrained(bert_model)
        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sent1 = str(self.data.loc[index, 'parent'])
        sent2 = str(self.data.loc[index, 'text'])
        
        #내가 새롭게 만든 거..tokenizer의 truncation이 잘 안 작동해서 manual하게 넣어주었음
   
        if len(sent1) > 300:
            sent1 = sent1[:300]
        if len(sent2) > 300:
            sent2 = sent2[:300]

        
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
      
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids

def build_dataloader(data, maxlen, batch_size, with_labels=True, bert_model='albert-base-v2'):
    dataset = CustomDataset(data, maxlen, bert_model)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    return data_loader

def main():
    parser = argparse.ArgumentParser(description="run this to build a dataloader for fine-tuning")
    parser.add_argument("--path_to_data", )
    parser.add_argument("--output-dir", default='./data',
                        help="Where to write out the dataloader")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Number of tokens per example")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--bert_model", default='albert-base-v2')
    args = parser.parse_args()
    data_loader = build_dataloader(data=args.path_to_data, maxlen=args.max_seq_lenth, batch_size=args.batch_size, bert_model=args.bert_model)
    

