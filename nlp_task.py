
#Library File
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import string
from string import punctuation
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

missing_values = ["n/a","na","--"]
data = pd.read_csv('company_data.csv', na_values = missing_values)
data.info()
sentiment_data_final = data[['management_attitude','Sentiment']]
sentiment_data_final_clean= sentiment_data_final.dropna()
sentiment_data_final_clean

def clean_text(text):
    """
    Make text lowercase, remove text in square brackets, remove links, remove punctuation
    and remove words containing numbers
    
    """
    text = text.lower()
    text = re.sub('\[.*?\]','', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text

df_sentiment = pd.DataFrame()
df_sentiment['attitude'] = sentiment_data_final_clean['management_attitude']
df_sentiment['sentiment'] = sentiment_data_final_clean['Sentiment']

df_sentiment['attitude'] = df_sentiment['attitude'].astype(str)
df_sentiment['sentiment'] = df_sentiment['sentiment'].astype(str)

df_sentiment['attitude'] = df_sentiment['attitude'].apply(lambda x: clean_text(x))
df_sentiment['sentiment'] = df_sentiment['sentiment'].apply(lambda x: x.replace("rt", ""))

df_sentiment.head()

def encode_labels(df):
    for i,j in df_sentiment.iterrows():
        if j['sentiment'] == 'Bad':
            j['sentiment'] = 0
        elif j['sentiment'] == 'Good':
            j['sentiment']=1
        elif j['sentiment']=='Moderate':
            j['sentiment']=2
    return df_sentiment

df_sentiment = encode_labels(df_sentiment)

df_sentiment['sentiment']= df_sentiment['sentiment'].astype(int)

df_sentiment.info()

#loading our BERT model
BERT_UNCASED = 'BERT_UNCASED'
#loading the pre-trained BertTokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_UNCASED)

# some basic operations to understand how BERT converts a sentence into tokens and then into IDs
sample_body = 'very nice behavior and tehy expectation is adequaate'
tokens = tokenizer.tokenize(sample_body)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_body}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')

# using encode_plus to add special tokens : [CLS]:101, [SEP]:102, [PAD]:0
encodings = tokenizer.encode_plus(
            sample_body,
            max_length=32,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
)

encodings.keys()

print('Input IDs : {}'.format(encodings['input_ids'][0]))
print('\nAttention Mask : {}'.format(encodings['attention_mask'][0]))

"""# Class for Dataset


"""

#setting maximum length of sentence
MAX_LENGTH = 150

class attitudes(Dataset):
    
    def __init__(self, attitude, label, tokenizer, max_len):
        self.attitude = attitude
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.attitude)
    
    def __getitem__(self, item):
        attitude = str(self.attitude[item])
        label = self.label[item]
        
        encoding = self.tokenizer.encode_plus(
        attitude,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')
        return {
        'attitude': attitude,
         'input_ids': encoding['input_ids'],
         'attention_mask': encoding['attention_mask'],
         'label': torch.tensor(label, dtype = torch.long)
          }

"""# Creating data loaders"""

from sklearn.model_selection import train_test_split

training_data, testing_data = train_test_split(
    df_sentiment,
    test_size = 0.1,
    random_state = RANDOM_SEED
)

testing_data, validation_data = train_test_split(
    testing_data,
    test_size=0.5,
    random_state=RANDOM_SEED
)

training_data.head(20)

def create_data_loader(data, tokenizer, max_len, batch_size):
    
    ds = attitudes(attitude=data.attitude.to_numpy(),
    label=data.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)
    
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4)


BATCH_SIZE = 64
train_data_loader = create_data_loader(training_data, tokenizer, MAX_LENGTH, BATCH_SIZE)
testing_data_loader = create_data_loader(testing_data, tokenizer, MAX_LENGTH, BATCH_SIZE)
val_data_loader = create_data_loader(validation_data, tokenizer, MAX_LENGTH, BATCH_SIZE)

df = next(iter(train_data_loader))
df.keys()

df['input_ids'].squeeze().shape

df['attention_mask'].squeeze().shape

df['label'].shape

print('managment_attitude  : ', df['attitude'][1])
print('input_ids : ', df['input_ids'].squeeze()[0])
print('attention_mask : ', df['attention_mask'].squeeze()[0])
print('label : ', df['label'][0])

from transformers import AutoTokenizer, AutoModelForMaskedLM
  
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

bert_model = AutoModelForMaskedLM.from_pretrained(model)

last_hidden_state, pooled_output = bert_model(
  input_ids=encodings['input_ids'],
  attention_mask=encodings['attention_mask'],
  return_dict = False   # this is needed to get a tensor as result
)

"""# Sentiment Classifier model"""

class SentimentClassifier(nn.Module):
    
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained(BERT_UNCASED)
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert_model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert_model(
        input_ids=input_ids,
        attention_mask = attention_mask
        )
        output = self.dropout(pooled_output)
        return self.out(output)

"""
label 0: Bad
label 1: Good
label 2: Moderate
"""
class_names = [0, 1, 2]
sentiment_classifier = SentimentClassifier(len(class_names))
sentiment_classifier = sentiment_classifier.to(device)