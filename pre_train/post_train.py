'''
_author:para
_ref:https://huggingface.co/transformers/training.html
https://huggingface.co/bert-base-uncased

'''

from dataset import PostTrainDataset
from transformers import BertModel, AutoTokenizer,BertForPreTraining,BertForSequenceClassification, Trainer, TrainingArguments

data_name = 'qa_pairs.txt'
root_path = 'data'

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

#model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



train_dataset = PostTrainDataset(data_name, 'train', root_path,'seq-classification')
val_dataset = PostTrainDataset(data_name, 'val', root_path, 'seq-classification')


training_args = TrainingArguments(
		    output_dir='./results',          # output directory
			    num_train_epochs=1,              # total # of training epochs
				    per_device_train_batch_size=32,  # batch size per device during training
					    per_device_eval_batch_size=64,   # batch size for evaluation
						    warmup_steps=500,                # number of warmup steps for learning rate scheduler
							    weight_decay=0.01,               # strength of weight decay
								    logging_dir='./logs',            # directory for storing logs
									)

trainer = Trainer(
		    model=model,                         # the instantiated 🤗 Transformers model to be trained
			    args=training_args,                  # training arguments, defined above
				    train_dataset=train_dataset,         # training dataset
					    eval_dataset=val_dataset            # evaluation dataset
						)

trainer.train()


trainer.save_model('results/cu-dstc-bert-base')
tokenizer.save_pretrained('results/cu-dstc-bert-base')
