import warnings

from metrics import *
import transformers
warnings.filterwarnings('ignore')

from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
import string
import pickle
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification, AutoTokenizer
from ast import literal_eval
from DatasetLoader import DatasetLoader
from sklearn.model_selection import train_test_split
from utils import *
from datetime import datetime
from const import *
import argparse



def predict_(arg_model):

    # Set a random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    if arg_model:
        model_path = arg_model
    else:
        print("You need to add a correct path to the model")
        return None
    
    MODEL = model_path.split("/")[-1]
    print("Model to predict is",MODEL)

    for path in [
        r"data/V2_test.pkl",
    ]:
        if "zero_shot" in path:
            print("\nrunning predictions for zero_shot data")
            output_path = "zero_shot"
        elif "test" in path:
            print("\nrunning predictions for test data")
            output_path = "test"
        else:
            print("\nrunning predictions for train data")
            output_path="train"
        with open(path, 'rb') as f:
            df=pickle.load(f).head(500)
        df = df[df["tags"].apply(lambda x : len(x) > 0 )]
        texts, tags = list(df["sentence"].values), list(df["tags"].values)

        # create the mapping tag <=> id
        unique_tags = sorted(set(tag for doc in tags for tag in doc))
        #with open(os.path.join('results-'+MODEL, 'tag2id.pkl'), 'rb') as f:
        #    tag2id = pickle.load(f)
        tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        id2tag = {id: tag for tag, id in tag2id.items()}
        # tokenize the word
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                    truncation=True, max_length=256)
        labeled_encodings = tokenize_and_align_labels(encodings, tags, tag2id)
        labels = labeled_encodings["labels"]
        dataset = DatasetLoader(encodings, labels)
        

        #MODEL = "distilbert" #VERSION + " model"
        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(unique_tags))

        model.eval()

        training_args = TrainingArguments(
            output_dir='./results-' + MODEL,  # output directory
            per_device_train_batch_size=100,  # batch size per device during training
            per_device_eval_batch_size=100,  # batch size for evaluation
        )
        trainer = Trainer(
            model=model,
            args=training_args)
        preds = trainer.predict(dataset)

        # LOG RESULTS
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('models/results-'+MODEL):
            os.makedirs('models/results-'+MODEL)
        pred_labels = np.argmax(preds.predictions, axis=2)
        
        # Convert tag IDs to actual tag names
        predictions_as_text = []
        for i in range(len(pred_labels)):
            word_ids = encodings.word_ids(batch_index=i)
            tag_sequence = []
            for word_id in word_ids:
                if word_id is None:
                    # Skip special tokens (e.g., [CLS], [SEP])
                    continue
                tag_id = pred_labels[i][word_id]
                tag_sequence.append(id2tag[tag_id])
            predictions_as_text.append(tag_sequence)

        # Combine with sentences for readability
        header = 4
        for sentence, predicted_tags in zip(texts, predictions_as_text):
            if header > 0:
                print(f"Sentence: {sentence}")
                print(f"Predicted Tags: {predicted_tags}")
                print("-" * 50)
            header -= 1
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Specify model path")
    args = parser.parse_args()

    model = args.model  # access the argument value using its name
    predict_(model)