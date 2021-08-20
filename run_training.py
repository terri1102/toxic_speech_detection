import argparse
import numpy as np
from tqdm import tqdm
import torch
import os
import random
from torch import nn 
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from model import SentencePairClassifier
from build_dataset import CustomDataset
from torch.utils.data import DataLoader, Dataset

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def evaluate_loss(net, device, criterion, dataloader):
    """ Evaluate loss during training """
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count

def train_bert(net, bert_model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):
    
    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits.squeeze(-1), labels.float())
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
              
                opti.zero_grad() #그래디언트 클리어


            running_loss += loss.item()
   
            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0


        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

      
        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    path_to_model='models/{}_lr_{}_val_loss_{}_ep_{}.pt'.format(bert_model, lr, round(best_loss, 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()



def main():
    #args : 
    parser = argparse.ArgumentParser(description="run this to build a dataloader for fine-tuning")
    parser.add_argument("--bert_model", default='albert-base-v2')
    parser.add_argument("--freeze_bert", default=False)
    parser.add_argument("--epochs", default=4)
    parser.add_argument("--lr", default=2e-5 )
    parser.add_argument("--iters_to_accumulate", default=2)
    args = parser.parse_args()


    set_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #load data_loaders
    print("loading train and validation data loaders")
    train_loader = torch.load('./data_loader/train_loader.pt')
    val_loader = torch.load('./data_loader/val_loader.pt')

    #Instantiate a model 
    net = SentencePairClassifier(args.bert_model, freeze_bert=args.freeze_bert)
    net.to(device)

    #Setting hyperparameters
    criterion = nn.BCEWithLogitsLoss()
    opti = AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    num_warmup_steps = 0 # The number of steps for the warmup phase.
    num_training_steps = args.epochs * len(train_loader)  # The total number of training steps
    t_total = (len(train_loader) // args.iters_to_accumulate) * args.epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    
    train_bert(net, args.bert_model, criterion, opti, args.lr, lr_scheduler, train_loader, val_loader, args.epochs, args.iters_to_accumulate)


if __name__ == '__main__':
    main()

#parameter 설정
#bert_model = 'albert-base-v2' 
#freeze_bert = False  # if True, freeze the encoder weights and only update the classification layer weights
#maxlen = 500  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
#bs = 16  # batch size
#iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
#lr = 2e-5  # learning rate
#epochs = 4  # number of training epochs