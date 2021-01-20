import torch

from transformers import BertTokenizer
from srl.inference.tokenize_doc import get_tokenized_doc
from srl.srl_model.controller import Controller
from srl.constants import LABELS


class Inference:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = Controller(dropout_rate=0.0, mlp_size=200).to(self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()  # Eval mode

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def perform_srl(self, doc):
        output_dict = get_tokenized_doc(doc["sentence"], self.tokenizer)
        output_dict['predicate'] = doc['predicate']

        argument_list = self.model(output_dict)
        predicate = doc["sentence"][doc["predicate"]]
        argument_list = [[token_idx, arg_idx, LABELS[arg_idx], predicate, doc["sentence"][token_idx]]
                         for token_idx, arg_idx in argument_list]
        return argument_list

