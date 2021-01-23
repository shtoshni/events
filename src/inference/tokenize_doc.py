from data_processing.kbp_2015.segment_preprocessed_files import split_into_segments
from data_processing.kbp_2015.segment_preprocessed_files import get_sentence_map, flatten, BERT_RE, NEWLINE_TOKEN
from kbp_2015_utils.constants import DOC_TYPES_TO_IDX


class DocumentState(object):
    def __init__(self, doc_type):
        self.doc_type = doc_type
        self.sentence_end = []
        self.token_end = []
        self.subtokens = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []

    def finalize(self):
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)
        # assert len(all_mentions) == len(set(all_mentions))
        num_words = len(flatten(self.segments))
        assert num_words == len(subtoken_map), (num_words, len(subtoken_map))
        assert num_words == len(sentence_map), (num_words, len(sentence_map))
        return {
            "doc_type": DOC_TYPES_TO_IDX[self.doc_type],
            "sentences": self.segments,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
        }


def get_tokenized_doc(doc, tokenizer, doc_type='newswire'):
    document_state = DocumentState(doc_type)

    tokenized_doc = []
    if isinstance(doc, str):
        tokenized_doc = tokenizer.tokenize(doc)
    elif isinstance(doc, list):
        tokenized_doc = []
        for sent in doc:
            tokenized_doc.extend(tokenizer.tokenize(sent))
            tokenized_doc.append(NEWLINE_TOKEN)

    word_idx = -1
    for idx, token in enumerate(tokenized_doc):
        if token == NEWLINE_TOKEN:
            # [NEWL] corresponds to "\n" in real doc
            document_state.sentence_end[-1] = True
            continue

        if not BERT_RE.match(token):
            word_idx += 1

        document_state.subtokens.append(token)
        if idx == len(tokenized_doc) - 1:
            # End of document
            document_state.token_end += ([True])
        else:
            next_token = tokenized_doc[idx + 1]
            if BERT_RE.match(next_token):
                # If the next token has ## at the start then the current subtoken
                # is clearly not the end of the token
                document_state.token_end += ([False])
            else:
                document_state.token_end += ([True])

        document_state.subtoken_map.append(word_idx)
        document_state.sentence_end.append(False)  # No info on sentence end

    split_into_segments(document_state, 512, document_state.sentence_end, document_state.token_end)
    document = document_state.finalize()
    return document


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    doc = "My father’s eyes had closed upon the light of this world six months, when Ishmael opened on it."
    print(get_tokenized_doc(doc, tokenizer))
