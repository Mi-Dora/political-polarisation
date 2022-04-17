from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd

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

def save(pred_biden, pred_trump, file):
    sentences = pd.read_csv(file).tolist()

    filename = file.split('/')[-1]
    base_url = './BERT/result/'
    result = []
    for i, text in enumerate(sentences):
        result.append(text + pred_biden[i] + pred_trump[i])
    name=['text',"Against Biden","Favor Biden","None Biden","Against Trump","Favor Trump","None Trump"]
    df = pd.DataFrame(columns=name,data=result)#数据有三列，列名分别为one,two,three
    df.to_csv(base_url+filename)

def predict(files, device):
    pretrained_LM_path_biden = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"
    tokenizer_biden = AutoTokenizer.from_pretrained(pretrained_LM_path_biden)
    model_biden = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path_biden).to(device)

    pretrained_LM_path_trump = "kornosk/bert-election2020-twitter-stance-trump-KE-MLM"
    tokenizer_trump = AutoTokenizer.from_pretrained(pretrained_LM_path_trump)
    model_trump = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path_trump).to(device)
    
    for file in files:
        sentences = pd.read_csv(file)['text'].tolist()
        sentences = [s.lower() for s in sentences]

        inputs = tokenizer_biden(sentences, return_tensors="pt").to(device)
        outputs = model_biden(**inputs).get('logits')
        pred_biden = torch.softmax(outputs, dim=1).detach().cpu().tolist()

        inputs = tokenizer_trump(sentences, return_tensors="pt").to(device)
        outputs = model_trump(**inputs).get('logits')
        pred_trump = torch.softmax(outputs, dim=1).detach().cpu().tolist()

        save(pred_biden, pred_trump, file)

# use pipline easily to implement
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# result = classifier(sentence)
# print('result', result)