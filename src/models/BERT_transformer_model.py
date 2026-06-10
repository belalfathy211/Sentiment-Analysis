import torch.nn as nn
from transformers import DistilBertModel


class BERTTransformerSentimentRNN(nn.Module):
    def __init__(self, dropout=0.3, output_dim=1):
        super(BERTTransformerSentimentRNN, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout= nn.Dropout(dropout)
        self.fc= nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.last_hidden_state[:,0,:]
        x = self.dropout(cls_output)
        return self.fc(x)