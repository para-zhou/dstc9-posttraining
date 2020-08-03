'''

_author:para


generate negative samples and labels
split final_combine.json into training and validation set

'''

import json
import random
from torch.utils.data import random_split
import os
data_path = 'final_combine.json'
data_name = 'qa_pairs.txt'

with open(data_path,'r') as f:
	data = json.load(f)

def sampling(data):
	sampled_data = []
	for d in data:
		q = d[0]
		a = d[1]
		false_a = a # random sample an answer and make sure it's not the true answer
		while false_a == a:
			false_a = random.sample(data,1)[0][1]
		
		
		sampled_data.append([q,a,'1'])
		sampled_data.append([q,false_a,'0'])
	print('done labeling')
	return sampled_data


def spliting(data):
	split_rate = int(len(data)/20) #(1/10 are validation set)
	random.seed(2020)
	random.shuffle(data)

	train_data = data[split_rate:]
	val_data = data[:split_rate]

	print('done spliting')
	data_writer(train_data,os.path.join('train', data_name))
	data_writer(val_data,os.path.join('val',data_name))
	data_writer(data,data_name)

def data_writer(data,path):
	with open(path,'w',encoding='utf-8',newline='') as f:
		for s in data:
			f.write('<s>'.join(s).replace('\n', ' ') +'\n')
	print('done writing %s' % path)
if __name__ == '__main__':
	spliting(sampling(data))





