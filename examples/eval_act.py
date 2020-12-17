from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging
import numpy as np

from preprocess_util import *
from sklearn.preprocessing import MultiLabelBinarizer
import random

random.seed(1234)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
with open("data/agent_intent_label.json") as in_file:
    label_list = json.load(in_file)
    label_list = [x for x in label_list.keys() if len(x) > 0]

label_bin = MultiLabelBinarizer(classes=label_list)
label_bin.fit([])

def read_data(input_file_name):
    data_raw = read_data_for_act(input_file_name)
    #data_raw = data_raw[:100]
    #random.shuffle(data_raw)

    text_a = [preprocess_history(x["history"]) for x in data_raw]
    #text_b = [preprocess_sys_usr(x["sys"], x["usr"]) for x in data_raw]
    sys_text = [x["sys"] for x in data_raw]
    label = label_bin.transform([x["act"] for x in data_raw])

    df = pd.DataFrame(zip(text_a, label), columns=["text", "labels"])
    print("Loaded %d data examples from %s" % (len(df), input_file_name))
    print(df.head())
    return df, sys_text

test_df, sys_text = read_data("data/test_with_act.json")

model_args = {
    'reprocess_input_data': True, 
    'max_seq_length': 512
}
# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel('bert', './outputs_act_new_label/best_model', use_cuda=True, args=model_args)

def _recall_topk(labels, preds, k):
    preds_argsort = (-preds).argsort()
    acc = 0
    for li, label in enumerate(labels):
        acc2 = 0
        count2 = 0
        for lj, l in enumerate(label):
            if l > 0:
                count2 += 1
                if lj in preds_argsort[li][:k]:
                    acc2 += 1
        if count2 > 0:
            acc += float(acc2) / count2
    acc = acc / len(labels)       
    return acc

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(
    test_df, 
    top1=lambda labels, preds: _recall_topk(labels, preds, 1), 
    top3=lambda labels, preds: _recall_topk(labels, preds, 3),
    top5=lambda labels, preds: _recall_topk(labels, preds, 5), 
    top10=lambda labels, preds: _recall_topk(labels, preds, 10)
)

print(result)
#print(model_outputs)
#print(wrong_predictions)

print("context\tpred\tactual\tactual_text")
for idx, x in test_df.iterrows():
    p = model_outputs[idx]
    predict_labels = np.where(p>0.15)
    predict_labels = [label_list[x] for x in predict_labels[0]]
    actual_labels = np.where(x["labels"]>0.5)
    actual_labels = [label_list[x] for x in actual_labels[0]]
    print("%s\t%s\t%s\t%s" % (x["text"], predict_labels, actual_labels, sys_text[idx]))


# Test on example
#test_example = ["[CLS] [SYS] hi. [USR] hi. [SYS]  [SEP] what's the dosage? [USR] 3 per day"]
#predictions, raw_outputs = model.predict(test_example)
#print(predictions)
#print(raw_outputs)

#preds = np.argmax(raw_outputs, axis=1)
#print("label:", label_list[preds[0]])

#print("text\tpred\tactual")
#for t,p in zip(test, preds):
#    print("%s\t%s\t%s" % (t[0], class_list[p], class_list[t[1]]))