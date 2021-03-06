import os
import json
import re
import string
import random

import numpy as np

from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from .utils import MyQADataset, MyDataLoader
from .zest_evaluate import evaluate_predictions

class ZESTData(object):

    def __init__(self, logger, args, data_path, is_training):
        self.data_path = data_path
        if args.debug:
            self.data_path = data_path.replace("train", "dev")
        with open(self.data_path, "r", encoding="utf-8") as f:
            json_list = list(f)

        if "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        elif "train" in self.data_path:
            self.data_type = "train"
        else:
            raise NotImplementedError()

        self.old_data = [json.loads(json_str) for json_str in json_list]
        self.data = []

        # following zest original code
        random.seed(5)

        for dp in self.old_data:
            for idx, example in enumerate(dp["examples"]):
                if self.data_type != "test":
                    answer = example["answer"]

                    # following zest original code
                    try:
                        # Special processing of multiple correct answers in structure formatted output.
                        # Chose one at random. Note the official eval script will
                        # consider all possible answers.
                        json_answer = json.loads(answer)
                        if isinstance(json_answer, list):
                            for row in json_answer:
                                for key in row.keys():
                                    value = row[key]
                                    if isinstance(value, list):
                                        value_choice = random.choice(value)
                                        row[key] = value_choice
                            answer = json.dumps(json_answer)
                    except (json.JSONDecodeError, TypeError):
                        pass

                    if isinstance(answer, list):
                        # Chose one at random.
                        answer_choice = random.choice(answer)
                        answer = answer_choice
                else:
                    answer = "TEST_NO_ANSWER"

                self.data.append({  "id": dp["id"],
                                    "in_task_id": idx,
                                    "question": dp["question"].replace("\n", " "),
                                    "context": example["context"].replace("\n", " "),
                                    "answer": answer
                                })

        if args.debug:
            self.data = self.data[:40]

        assert type(self.data)==list
        assert all(["id" in d for d in self.data]), self.data[0].keys()

        if type(self.data[0]["id"])==int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])  

        self.index2id = {i:d["id"] for i, d in enumerate(self.data)}
        self.id2index = {d["id"]:i for i, d in enumerate(self.data)}
        
        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args

        self.metric = "F1"
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.gen_early_stop = False

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        # not sure what this means
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(".json", "-{}.json".format(postfix)))
        
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata = json.load(f)

        else:
            print("Start tokenizing ... {} instances".format(len(self.data)))

            questions = []
            answers = []

            for dp in self.data:
                questions.append(" zest question: {} zest context: {}".format(dp["question"], dp["context"]))
                answers.append([dp["answer"]])

            print("Printing Examples ...")
            for i in range(3):
                print(questions[i])
                print(answers[i])
                print()

            answers, metadata = self.flatten(answers) # what is metadata?

            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]
            if self.args.append_another_bos:
                questions = ["<s> "+question for question in questions]
                answers = ["<s> " +answer for answer in answers]
            
            print("Tokenizing Input ...")
            question_input = tokenizer.batch_encode_plus(questions,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            print("Tokenizing Output ...")
            answer_input = tokenizer.batch_encode_plus(answers,
                                                       pad_to_max_length=True,
                                                       max_length=self.args.max_output_length)

            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            if self.load:
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     metadata]
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata], f)

        self.dataset = MyQADataset(input_ids, attention_mask,
                                        decoder_input_ids, decoder_attention_mask,
                                        in_metadata=None, out_metadata=metadata,
                                        is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, verbose=False):
        if self.data_type == 'test':
            return (-1.0, -1.0, -1.0)
        dev = []
        with open(self.data_path, "r") as fin:
            for line in fin:
                dev.append(json.loads(line))

        processed_predictions = []
        for pred in predictions:
            pred = pred.strip()

            # if an empty string is predicted, set it to "n/a"
            if len(pred) == 0:
                pred = "n/a"

            if pred[0] == '"' and pred[-1] == '"':
                pred = json.loads(pred)
            processed_predictions.append(pred)

        score = evaluate_predictions(dev, processed_predictions, os.path.join(self.args.output_dir, "results.json"), verbose)
        # f1s = []

        # for i in range(5):
        #     print(predictions[i])
        #     print(self.raw_questions[i])
        #     print(self.raw_answers[i])

        # for (prediction, dp) in zip(predictions, self.raw_answers):
        #     f1s.append(get_f1_over_list(prediction.strip(), dp))
        return score

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))

        predictions = ['n/a' if len(prediction.strip())==0 else prediction for prediction in predictions]
        prediction_text = [prediction.strip()+'\n' for prediction in predictions]
        save_path = os.path.join(self.args.output_dir, "{}_predictions.txt".format(self.args.prefix))
        with open(save_path, "w") as f:
            f.writelines(prediction_text)
        
        self.logger.info("Saved prediction in {}".format(save_path))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([f1_score(prediction, gt) for gt in groundtruth])
    return f1_score(prediction, groundtruth)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

