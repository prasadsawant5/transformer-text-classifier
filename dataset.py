import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict

from config import DATA, EOS, PAD, PATHS

def get_label_from_one_hot(arr: np.array) -> str:
    idx = np.argmax(arr)

    if idx == 0:
        return 'positive'
    elif idx == 1:
        return 'neutral'
    else:
        return 'negative'
    
def get_sentence_from_vec(vec: List[int]) -> str:
    res = ''
    with open(f'./{DATA}/word_tokens.json', 'r') as f:
        word_to_vec = json.load(f)
        
        for ve in vec:
            for k, v in word_to_vec.items():
                if v == ve:
                    res += k
                    res += ' '
                    break
    return res


def one_hot_label(label: str) -> np.array:
    arr = np.zeros((3,), dtype=np.float16)
    if label == 'positive' or 'positive' in label:
        arr[0] = 1.
    elif label == 'neutral' or 'neutral' in label:
        arr[1] = 1.
    else:
        arr[-1] = 1.

    return arr

def tokenize_sentence(sentence: str, word_to_vec: Dict) -> List[int]:
    arr = []
    for word in sentence.split(' '):
        arr.append(word_to_vec[word])
    return arr

def export_to_csv() -> int:
    sentences = set()
    data = []
    prompts = []
    labels = []

    for _path in PATHS:
        with open(_path, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                if line not in sentences:
                    sentences.add(line)
                    sentence = line.strip().lower()
                    if '.@' in sentence:
                        prompt = sentence.split('.@')[0]
                        # prompt += EOS if prompt[-1] == ' ' else f' {EOS}'
                        prompts.append(prompt)
                        label = sentence.split('.@')[-1]
                        labels.append(label)
                    else:
                        prompt = sentence.split('@')[0]
                        # prompt += EOS if prompt[-1] == ' ' else f' {EOS}'
                        prompts.append(prompt)
                        label = sentence.split('@')[-1]
                        labels.append(label)

                    data.append([prompt, label])

    MAX_TOKENS = len(max(prompts, key=len)) + 1

    padded_sentences = []
    for prompt in prompts:
        if len(prompt.split()) < MAX_TOKENS:
            for i in range(len(prompt.split()), MAX_TOKENS - 1):
                prompt += PAD if prompt[-1] == ' ' else f' {PAD}'
            prompt += EOS if prompt[-1] == ' ' else f' {EOS}'
            if len(prompt.split()) != MAX_TOKENS:
                print(len(prompt.split()))
        padded_sentences.append(prompt)

    df = pd.DataFrame(data={'sentence': padded_sentences, 'label': labels}, index=None)
    df.to_csv(f'./{DATA}/financial_data.csv')

    return MAX_TOKENS

def generate_word_to_vec() -> int:
    df = pd.read_csv(f'./{DATA}/financial_data.csv')
    sentences = df['sentence'].tolist()
    

    words = set()
    for sentence in sentences:
        for word in sentence.split():
            words.add(word)

    VOCAB_SIZE = len(words)

    word_tokens = {}
    for i, word in enumerate(words):
        word_tokens[word] = i
        
    with open(f'./{DATA}/word_tokens.json', 'w') as f:
        json.dump(word_tokens, f)

    return VOCAB_SIZE

def calculate_class_weights(df: pd.DataFrame) -> Dict:
    weights = {0: 0, 1: 0, 2: 0}
    labels = df['label'].tolist()

    for l in labels:
        if 'positive' in l:
            weights[0] = weights[0] + 1
        elif 'neutral' in l:
            weights[1] = weights[1] + 1
        else:
            weights[2] = weights[2] + 1

    print(f'Weightage: {weights}')
    return weights

def get_dataset() -> tuple:
    df = pd.read_csv(f'./{DATA}/financial_data.csv')
    weights = calculate_class_weights(df)

    features = []
    labels = []

    with open(f'./{DATA}/word_tokens.json', 'r') as f:
        word_to_vec = json.load(f)

        for _, row in df.iterrows():
            sentence = row['sentence']
            label = row['label']

            features.append(tokenize_sentence(sentence, word_to_vec))
            labels.append(one_hot_label(label))

    return features, labels, weights