from srl.data_processing.conll09_to_json import tokenize_sentence


def get_tokenized_doc(doc, tokenizer):
    tokenized_sentence, subtoken_map = tokenize_sentence(doc, tokenizer)

    return {
        'sentences': [tokenized_sentence],
        'start_idx': [start_idx for start_idx, end_idx in subtoken_map],
        'end_idx': [end_idx for start_idx, end_idx in subtoken_map],
    }

