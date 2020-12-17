
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import json
import numpy as np

from preprocess_util import *


logging.basicConfig(level=logging.ERROR)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

with open("data/sentence_list_small.json") as in_file:
    sentence_list = json.load(in_file)
    sentence_list = [x for x in sentence_list if "@" not in x]
    sentence_list = [x for x in sentence_list if 25 < len(x) < 150]


history = [
    "this is john. how may I help you today?",
    "i got a letter in mail, about medicare.",
    "i can definitely help you on that. can i have your name please?",
    "my name is tom. t o m"
]

history = [
    "how many milligrams?",
    "50 milligrams",
    "and how many times a day?",
    "three"
]

history = [
    "yeah. i don't think like that.",
    "i'm just thinking, you know, i'm trying to take care of my diabetes"
]

test_data = []
text_a = preprocess_history(history)
for sentence in sentence_list:
    test_data.append((text_a, sentence, 0))

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


assert len(model_outputs) == len(test_data)

arr_0 = np.array([x[0] for x in model_outputs])
arr = np.array([x[1] for x in model_outputs])
max_result = (arr.argsort()[-50:][::-1])

print("Predictions:")
for m in max_result:
    print("  ", sentence_list[m], arr_0[m], arr[m])

