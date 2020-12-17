from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import numpy as np

from preprocess_util import *

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

all_data = read_sentences("data/merged_transcript_classified.json")

# Create a ClassificationModel
model = ClassificationModel('bert', './outputs_sent_ranking_with_tod/best_model')





def output_sent_ranking(sentences):
    predictions, raw_outputs = model.predict(sentences)

    outputs_positive = [x[1] for x in raw_outputs]
    outputs = list(zip(sentences, outputs_positive))
    outputs.sort(key=lambda x:x[1], reverse=True)

    for o in outputs[:10]:
        print(" ", o[0], o[1])


for data in [all_data[200], all_data[300]]:

    agent_sentences = data["agent"]
    customer_sentences = data["customer"]

    print("Customer:")
    output_sent_ranking(customer_sentences)
    print("Agent:")
    output_sent_ranking(agent_sentences)
