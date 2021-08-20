from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AlbertTokenizerFast
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import torch
import pandas as pd
import glob
import argparse

class CustomDataset(Dataset):
    """
    CustomDataset class makes customdataset and data loader from a csv file. The default location for the data file is $/data.
    This class takes two sentences seperately and concatenates them with special tokens. 
    """
    
    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):

        self.data = data  
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  
        #self.tokenizer = AlbertTokenizerFast.from_pretrained(bert_model)
        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sent1 = str(self.data.loc[index, 'parent'])
        sent2 = str(self.data.loc[index, 'text'])
        
        #내가 새롭게 만든 거..tokenizer의 truncation이 잘 안 작동해서 manual하게 넣어주었음
   
        if len(sent1) > 1000:
            sent1 = sent1[:1000]
        if len(sent2) > 1000:
            sent2 = sent2[:1000]

        
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

def split_dataset(data):
    train, test = train_test_split(data, test_size=0.2, random_state=1, stratify=data['label'])
    val, test = train_test_split(test, test_size=0.5, random_state=1, stratify=test['label'])
    print(train.shape, val.shape, test.shape)
    return train, val, test

def build_dataloader(data, maxlen, batch_size, with_labels=True, bert_model='albert-base-v2'):
    "Creates Pytorch DataLoaders from CustomDataset"
    dataset = CustomDataset(data, maxlen, bert_model)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    return data_loader

def main():
    parser = argparse.ArgumentParser(description="run this to build a dataloader for fine-tuning")
    parser.add_argument("--path_to_data", default='./data')
   # parser.add_argument("--output-dir", default='./data_loader',
   #                     help="Where to write out the dataloader")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Number of tokens per example")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--bert_model", default='albert-base-v2')
    args = parser.parse_args()

    data = os.path.join(args.path_to_data, 'final_data.csv')
    data = pd.read_csv(data)
    data = data.fillna("")
    data = data[['index','parent','text','label']]

   # train, val = train_test_split(data, test_size=0.2, random_state=1, stratify=data['label'])
   # val, test = train_test_split(val, test_size=0.5, random_state=1, stratify=val['label'])
  
    train, val, test = split_dataset(data)
    
    train_loader = build_dataloader(data=train, maxlen=args.max_seq_length, batch_size=args.batch_size, bert_model=args.bert_model)
    val_loader = build_dataloader(data=val, maxlen=args.max_seq_length, batch_size=args.batch_size, bert_model=args.bert_model)
    test_loader = build_dataloader(data=test, maxlen=args.max_seq_length, batch_size=args.batch_size, bert_model=args.bert_model)

    torch.save(train_loader, './data_loader/train_loader.pt')
    torch.save(val_loader, './data_loader/val_loader.pt')
    torch.save(test_loader, './data_loader/test_loader.pt')
    
    #return train_loader, val_loader, test_loader

if __name__ == "__main__":
    main()

    

