from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import re
import math 
from transformers import pipeline
import os
import random
import numpy as np

from nlp import load_dataset

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, print_text=False):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path
        self.ssm = False
        if self.args.dataset == 'recentnews':
            if type_path=='train':
                self.dataset = pd.read_csv('data/recent_news_20000.csv')
            else:
                self.dataset = self.get_recent_val(-1,-1) #Getting validation data for both LAMA-entity and RecentProbe
        else:
            raise NameError('Select the correct Dataset!')
        print(f'length of dataset: {len(self.dataset)}')
        if self.args.dataset == 'recentnews' and type_path=='validation':
            self.input_length = 50
            self.output_length = 4
        else:
            self.input_length = input_length
            self.output_length = output_length
        sentinels=[]
        for i in range(100):
            sentinels.append(f'<extra_id_{i}>')
        self.sentinels = sentinels
        
    def get_recent_val(self, lama_num, recent_num):
        lama = pd.read_csv('data/lama_template.csv')
        recent = pd.read_csv('data/recent_news_summary_6000.csv')
        dataset = []
        for index, row in lama.iterrows():
            dataset.append(row)
        for index, row in recent.iterrows():
            dataset.append(row)
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        text = re.sub("\\[.*\\]",'',text)
        return text

    def convert_to_features(self, example_batch):
        if self.args.dataset == 'recentnews':
            input_ = example_batch['input']
            target_ = example_batch['output']
            if type(target_)!=str:
                target_=''
        else:
            raise Exception('Select the correct dataset!')
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        return source, targets
  
    def __getitem__(self, index):
        if self.type_path=='train':
            source, targets = self.convert_to_features(self.dataset.iloc[index])
        else:
            source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}