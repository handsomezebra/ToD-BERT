import random

from preprocess_util import *

random.seed(1234)

def norm_text(text):
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

def read_write_data(input_file_name, output_file_name):
    data_raw, sys_response_sample = read_data_for_rs(input_file_name)
    random.shuffle(data_raw)
    random.shuffle(sys_response_sample)

    data = []

    for x in data_raw:
        text_a = preprocess_history(x["history"])
        text_b = x["sys"]
        data.append((text_a, "[SYS] " + text_b, 1))
        for neg_response in random.sample(sys_response_sample, 9):
            while neg_response == text_b:
                print("Conflict!!!")
                neg_response = random.choice(sys_response_sample)
            data.append((text_a, "[SYS] " + neg_response, 0))

    random.shuffle(data)
    print("Loaded %d data examles from %s" % (len(data), input_file_name))

    with open(output_file_name, "w") as out_file:
        for text_a, text_b, label in data:
            text_a = norm_text(text_a)
            text_b = norm_text(text_b)
            out_file.write(text_a + "\t" + text_b + "\t" + str(label) + "\n")

read_write_data("data/train.json", "data/train.tsv")

read_write_data("data/dev.json", "data/dev.tsv")

read_write_data("data/test.json", "data/test.tsv")
