import numpy as np
import torch
import torch.nn as nn
import transformers as trf
import nlp
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.ids = dataframe.ids
        self.targets = dataframe.list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return (
            torch.tensor(self.ids[index], dtype=torch.long),        #ids
            torch.tensor(self.targets[index], dtype=torch.float)    #label
        )

class albert_1024(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feed_forward = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*312, 256),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, 18),
#            nn.Sigmoid()
        )
    def forward(self, x):
        x1, x2 = torch.split(x, 512, dim=1)
        return self.feed_forward(self.model(x1).last_hidden_state.view(-1, 512*312)) + self.feed_forward(self.model(x2).last_hidden_state.view(-1, 512*312))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    """
    df = pd.read_csv("./dev.csv")
    df = df.replace({1: 1, -1: 1, 0: 1, -2: 0}).fillna(0)
    df['list'] = df[df.columns[2:]].values.tolist()
    """
    #model_name = "WENGSYX/Deberta-Chinese-Large"
    #model_name = "ckiplab/albert-base-chinese"
    model_name = "ckiplab/albert-tiny-chinese"
    """
    tokenizer = trf.BertTokenizer.from_pretrained(model_name)
    embedded = tokenizer.batch_encode_plus(
            df['review'].tolist(),
            max_length=1024,
            #truncation=True,
            padding='max_length',
            #stride=3,
            #return_overflowing_tokens=True,
            #return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=False,
            #return_length=True,
        )
    #print(max([len(x) for x in embedded['input_ids']]))
    for x in embedded['input_ids']:
        print(', '.join([str(y) for y in x]))
    exit()
    """
    config = trf.PretrainedConfig.from_pretrained(model_name).update({
                                #'problem_type': 'multi_label_classification',
                                'model_type': 'AlbertConfig',
                                'num_labels': 18})
    #model = trf.BertForSequenceClassification.from_pretrained(model_name, config=config)
    model = trf.AutoModel.from_pretrained(model_name)
    #bert = trf.AutoModel.from_pretrained(model_name)
    #classifier = trf.AutoModelForSequenceClassification.from_config(config)
    #classifier.bert = bert
    model = albert_1024(model)
    model.to(torch.device("cuda:0"))
    model.train()

    ids = [[int(y) for y in x.rstrip().split(',')] for x in open("ckipl_abalbert-base-chinese_tk.txt").readlines()]
    df = pd.read_csv("./dev.csv").drop(columns=['id', 'review'])
    df = df.replace({1: 1, -1: 1, 0: 1, -2: 0}).fillna(0)
    df['list'] = df[df.columns[0:]].values.tolist()
    df['ids'] = ids
    del ids
    df = df[['ids', 'list']]

    training_data = CustomDataset(df)
    train_loader = DataLoader(training_data, batch_size=12, shuffle=True)

    from tqdm import tqdm
    optimizer = torch.optim.AdamW(params = model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    for epoch in range(3):
        t = tqdm(train_loader)
        for _, x in enumerate(t, 0):
            outputs = model(x[0].to(torch.device("cuda:0")))
            label = x[1].to(torch.device("cuda:0"))

            optimizer.zero_grad()
            #loss = nn.BCELoss()(outputs, x[1])
            loss = nn.BCEWithLogitsLoss()(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item())

    torch.save(model, "model")