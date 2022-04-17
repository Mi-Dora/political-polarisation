from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

id2label = {
    0: "Against Trump",
    1: "Favor Trump",
    2: "NONE Trump"
}

def main(paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-trump-KE-MLM"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path).to(device)

    # predict
    sentences = ['This pandemic has shown us clearly the vulgarity of our healthcare system. Highest costs in the world, yet not enough nurses or doctors. Many millions uninsured, while insurance company profits soar. The struggle continues. Healthcare is a human right. Medicare for all.', 'This pandemic has shown us clearly the vulgarity of our healthcare system. Highest costs in the world, yet not enough nurses or doctors. Many millions uninsured, while insurance company profits soar. The struggle continues. Healthcare is a human right. Medicare for all.']
    inputs = tokenizer([s.lower() for s in sentences], return_tensors="pt").to(device)
    outputs = model(**inputs).get('logits')
    predicted_probability = torch.softmax(outputs, dim=1).tolist()
    print('predicted_probability: ', predicted_probability)


# use pipline easily to implement
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# result = classifier(sentence)
# print('result', result)