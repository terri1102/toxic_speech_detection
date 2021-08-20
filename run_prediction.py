import torch
from model import SentencePairClassifier
import os
import argparse

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function.
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file.
    """
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(prob)+'\n' for prob in probs_all)
    w.close()

if __name__ == '__main__':
    #argument parser
    parser = argparse.ArgumentParser(description="run this to build a dataloader for fine-tuning")
    parser.add_argument("--bert_model", default='albert-base-v2')

    args = parser.parse_args()

    path_to_model = os.path.join('./models','*/.pt')
    model = SentencePairClassifier(args.bert_model)
    test_loader = torch.load('./data_loader/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print()

    #Loading the fine-tunned weights to the model
    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    print("Predicting on test data...")
    path_to_output_file = './results/output.txt' #

    test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                    result_file=path_to_output_file)
    print()

    print("Predictions are available in : {}".format(path_to_output_file))