import torch

from torch.types import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

class LLMHandler:
    def __init__(self, model_name: str):
        self.__device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.__tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(model_name)
        self.__model: BertModel = AutoModel.from_pretrained(model_name).to(self.__device)

    def get_classification_embedding(self, text: str | list[str]) -> Tensor:
        inputs = self.__tokenizer(text, return_tensors="pt", padding=True).to(self.__device)

        with torch.no_grad():
            outputs = self.__model(**inputs)

        return outputs.last_hidden_state[:, 0, :]
