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
        sentences = [s.lower() for s in sentences]
        self.sentence_list = sentences
 
    def __getitem__(self, index):
        return self.sentence_list[index]
 
    def __len__(self):
        return len(self.sentence_list)

def save(pred_biden, pred_trump, file):
    filename = file.split('/')[-1]
    base_url = './BERT/result/'

    df = pd.read_csv(file)
    df_biden = DataFrame(pred_biden, columns=["Against Biden","Favor Biden","None Biden"])
    df_trump = DataFrame(pred_trump, columns=["Against Trump","Favor Trump","None Trump"])
    df = pd.concat([df,df_biden,df_trump], axis=1)
    print('new data: ', df.head())
    
    if os.path.exists(base_url+filename):
        df_existed = pd.read_csv(base_url+filename)
        df = pd.concat([df_existed, df], ignore_index=True)
    print('all data: ', df.head())
    df.to_csv(base_url+filename)

def predict(files, device):
    max_len = 128
    batch_size=128

    pretrained_LM_path_biden = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"
    tokenizer_biden = AutoTokenizer.from_pretrained(pretrained_LM_path_biden)
    model_biden = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path_biden).to(device)

    pretrained_LM_path_trump = "kornosk/bert-election2020-twitter-stance-trump-KE-MLM"
    tokenizer_trump = AutoTokenizer.from_pretrained(pretrained_LM_path_trump)
    model_trump = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path_trump).to(device)
    
    for file in files:
        dataset = MyDataSet(file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for sentences in tqdm(dataloader):
            inputs = tokenizer_biden(sentences, return_tensors="pt",padding='max_length',truncation=True,max_length=max_len).to(device)
            outputs = model_biden(**inputs).get('logits')
            pred_biden = torch.softmax(outputs, dim=1).detach().cpu().tolist()

            inputs = tokenizer_trump(sentences, return_tensors="pt",padding='max_length',truncation=True,max_length=max_len).to(device)
            outputs = model_trump(**inputs).get('logits')
            pred_trump = torch.softmax(outputs, dim=1).detach().cpu().tolist()

            save(pred_biden, pred_trump, file)

# use pipline easily to implement
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# result = classifier(sentence)
# print('result', result)