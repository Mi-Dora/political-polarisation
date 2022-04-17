from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

id2label = {
    0: "Against Biden",
    1: "Favor Biden",
    2: "NONE Biden"
}

# TODO load the data
def predict(paths, device):
    pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path).to(device)

    # predict
    sentences = ['This pandemic has shown us clearly the vulgarity of our healthcare system. Highest costs in the world, yet not enough nurses or doctors. Many millions uninsured, while insurance company profits soar. The struggle continues. Healthcare is a human right. Medicare for all.', 'This pandemic has shown us clearly the vulgarity of our healthcare system. Highest costs in the world, yet not enough nurses or doctors. Many millions uninsured, while insurance company profits soar. The struggle continues. Healthcare is a human right. Medicare for all.']
    inputs = tokenizer([s.lower() for s in sentences], return_tensors="pt").to(device)
    outputs = model(**inputs).get('logits')
    predicted_probability = torch.softmax(outputs, dim=1).tolist()

    return predicted_probability

predict([])

# use pipline easily to implement
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# result = classifier(sentence)
# print('result', result)