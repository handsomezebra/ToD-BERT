
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import json
import numpy as np

from preprocess_util import *


logging.basicConfig(level=logging.ERROR)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

with open("data/intent_list.json") as in_file:
    intent_list = json.load(in_file)

intent_label_list = [x[0] for x in intent_list]
intent_utterance_list = [x[1] for x in intent_list]
intent_label_to_utterance = {x[0] : x[1] for x in intent_list}

input_file_name = "data/test_with_act.json"
test_data_raw = read_data_for_rs_intent(input_file_name)
test_data_raw = test_data_raw[:1000]
print("Loaded %d data examples from %s" % (len(test_data_raw), input_file_name))

test_data = []

nb_example_per_test = len(intent_utterance_list)

for x in test_data_raw:
    text_a = preprocess_history(x["history"])
    assert len(x["act"]) > 0
    act = x["act"][0]
    correct_intent_utterance = intent_label_to_utterance[act]
    for intent_utt in intent_utterance_list:
        if intent_utt != correct_intent_utterance:
            test_data.append((text_a, intent_utt, 0))
        else:
            test_data.append((text_a, intent_utt, 1))

assert len(test_data) == len(test_data_raw) * nb_example_per_test

test_df = pd.DataFrame(test_data, columns=['text_a', 'text_b', 'labels'])
print("test_data length", len(test_df))
print(test_df.head())

model_args = {
    'reprocess_input_data': True, 
    'max_seq_length': 512,
    'lazy_loading': False
}

# Create a ClassificationModel
model = ClassificationModel('bert', './outputs_rs/best_model', use_cuda=True, args=model_args)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)
print(result)
#print(model_outputs)
#print(wrong_predictions)


assert len(model_outputs) == len(test_data)
r_list = list(zip(test_data, model_outputs))

# group the list by the same context
# each group is 157 samples
r_list_new = [r_list[i:i+nb_example_per_test] for i in range(0, len(r_list), nb_example_per_test)]
assert len(r_list_new) == len(test_data_raw)
for idx, r in enumerate(r_list_new):
    context = r[0][0][0]
    utterance = r[0][0][1]

    arr_0 = np.array([x[1][0] for x in r])
    arr = np.array([x[1][1] for x in r])
    max_result = (arr.argsort()[-3:][::-1])

    correct_act = test_data_raw[idx]["act"]

    print(context)
    print("Label:")
    print("  ", correct_act)
    print("Predictions:")
    for m in max_result:
        print("  ", intent_label_list[m], arr_0[m], arr[m])

    print()
