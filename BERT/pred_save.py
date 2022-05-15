from lib2to3.pgen2.tokenize import tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

id2label_biden = {
    0: "Against Biden",
    1: "Favor Biden",
    2: "None Biden"
}

id2label_trump = {
    0: "Against Trump",
    1: "Favor Trump",
    2: "None Trump"
}

class MyDataSet(Dataset):
    def __init__(self, file):
        sentences = pd.read_csv(file)['text'].tolist()
        self.sentence_list = sentences

    def __getitem__(self, index):
        return self.sentence_list[index]
 
    def __len__(self):
        return len(self.sentence_list)

def predict(files, device):
    max_len = 128
    batch_size=128

    pretrained_LM_path_biden = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"
    tokenizer_biden = AutoTokenizer.from_pretrained(pretrained_LM_path_biden)
    model_biden = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path_biden).to(device)

    pretrained_LM_path_trump = "kornosk/bert-election2020-twitter-stance-trump-KE-MLM"
    tokenizer_trump = AutoTokenizer.from_pretrained(pretrained_LM_path_trump)
    model_trump = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path_trump).to(device)
    base_url = './BERT/result/'
    
    for file in files:
        dataset = MyDataSet(file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds_biden = []
        preds_trump = []
        filename = file.split('/')[-1]
        for o_sentences in tqdm(dataloader):
            inputs = tokenizer_biden(o_sentences, return_tensors="pt",padding='max_length',truncation=True,max_length=max_len).to(device)
            outputs = model_biden(**inputs).get('logits')
            pred_biden = torch.softmax(outputs, dim=1).detach().cpu().tolist()
            
            inputs = tokenizer_trump(o_sentences, return_tensors="pt",padding='max_length',truncation=True,max_length=max_len).to(device)
            outputs = model_trump(**inputs).get('logits')
            pred_trump = torch.softmax(outputs, dim=1).detach().cpu().tolist()

            preds_biden += pred_biden
            preds_trump += pred_trump

        df = pd.read_csv(file)
        df_biden = DataFrame(preds_biden, columns=["Against Biden","Favor Biden","None Biden"])
        df_trump = DataFrame(preds_trump, columns=["Against Trump","Favor Trump","None Trump"])
        df_res = pd.concat([df,df_biden,df_trump], axis=1, sort=False)

        df_res.to_csv(base_url+filename)


# def pred2(files, device):
#     # use pipline easily to implement
#     pretrained_LM_path_biden = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"
#     pretrained_LM_path_trump = "kornosk/bert-election2020-twitter-stance-trump-KE-MLM"
#     tokenizer_biden = AutoTokenizer.from_pretrained(pretrained_LM_path_biden)
#     # tokenizer_trump = AutoTokenizer.from_pretrained(pretrained_LM_path_trump)


#     classifier_biden = pipeline('sentiment-analysis', model=pretrained_LM_path_biden, tokenizer=tokenizer_biden)
#     # classifier_trmp = pipeline('sentiment-analysis', model=pretrained_LM_path_trump, tokenizer=tokenizer_trump)
#     result = classifier_biden(" We're all going to get crack and $$$ from China and the Ukraine just like his own family!!!!  Vote !!!!!")
#     print('result', result)

# pred2(1, 2)