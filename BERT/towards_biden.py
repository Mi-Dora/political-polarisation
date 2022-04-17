from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

# choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# select mode path here
pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden-KE-MLM"

# load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)

id2label = {
    0: "Against Biden",
    1: "Favor Biden",
    2: "NONE Biden"
}

sentences = ['This pandemic has shown us clearly the vulgarity of our healthcare system. Highest costs in the world, yet not enough nurses or doctors. Many millions uninsured, while insurance company profits soar. The struggle continues. Healthcare is a human right. Medicare for all.', 'This pandemic has shown us clearly the vulgarity of our healthcare system. Highest costs in the world, yet not enough nurses or doctors. Many millions uninsured, while insurance company profits soar. The struggle continues. Healthcare is a human right. Medicare for all.']
inputs = tokenizer([s.lower() for s in sentences], return_tensors="pt")
outputs = model(**inputs).get('logits')
predicted_probability = torch.softmax(outputs, dim=1).tolist()
print('predicted_probability: ', predicted_probability)


# print("Prediction:", id2label[np.argmax(predicted_probability)])

# use pipline easily to implement
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# result = classifier(sentence)
# print('result', result)