'''
_author: para
_ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://github.com/para-zhou/DSTC9/blob/master/baseline/dataset.py
https://mccormickml.com/2019/07/22/BERT-fine-tuning/

'''

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from transformers import BertTokenizer




data_name = 'qa_pairs.txt'



class PostTrainDataset(Dataset):
	def __init__(self,data_name,mod, root_dir =None,model_type='Bert'):
		
		self.path = os.path.join(root_dir, mod, data_name)
		self.data = open(self.path,'r',encoding='utf-8').readlines()
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.model_type = model_type


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		d = self.data[idx].strip('\n').split('<s>')
		text = '%s [SEP] %s' %(d[0],d[1])
		encoding = self.tokenizer(text, 
				add_special_tokens=True, 
				padding='max_length',
				truncation = True,
				max_length=128,	
				return_attention_mask=True, 
				return_tensors = 'pt')
		#bert_input = '[CLP] %s [SEP] %s [SEP]' %(d[0],d[1])
		#print:wq：(d)
		#print(d[2])
		label = [0,0]

		label[ 1 - int(d[2]) ] = 1 # 1-> [1,0]; 0=> [0,1]
		label = torch.tensor(label).squeeze(0)
		label = torch.tensor(int(d[2]))
		#print(label)
		sample = {'input_ids':encoding['input_ids'][0],'attention_mask':encoding['attention_mask'][0]}
		if self.model_type == 'seq-classification':
			sample = {'input_ids':encoding['input_ids'][0], 
				'attention_mask':encoding['attention_mask'][0],
				'labels':label}

		#print(i, sample['input_ids'].size(), sample['attention_mask'].size())#, sample['labels'].size())
		#print(sample['input_ids'])
		return sample

train_dataset = PostTrainDataset(data_name,'train','data')
for i in range(5):
	sample = train_dataset[i]
	#print(i, sample['input_ids'].size(), sample['attention_mask'].size(), sample['labels'].size())
