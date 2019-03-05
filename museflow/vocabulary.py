class Vocabulary:
    """Takes care of converting back and forth between tokens and integer IDs."""

    def __init__(self, wordlist, pad_token='<pad>', start_token='<s>', end_token='</s>'):
        wordlist = list(wordlist)

        if pad_token:
            if pad_token not in wordlist:
                wordlist.insert(0, pad_token)
            elif wordlist[0] != pad_token:
                raise ValueError('<pad> has index {}, expected 0'.format(wordlist.index(pad_token)))

        if start_token and start_token not in wordlist:
            wordlist.append(start_token)
        if end_token and end_token not in wordlist:
            wordlist.append(end_token)

        if len(set(wordlist)) != len(wordlist):
            raise ValueError('Vocabulary has duplicate items')

        self._id2token = wordlist
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token

        self._token2id = {val: i for i, val in enumerate(wordlist)}
        self.pad_id = self._token2id.get(pad_token)
        self.start_id = self._token2id.get(start_token)
        self.end_id = self._token2id.get(end_token)

    def __iter__(self):
        return iter(self._id2token)

    def __get__(self, key, owner):
        return self._token2id[key]

    def __len__(self):
        return len(self._id2token)

    def __contains__(self, item):
        return item in self._id2token

    def to_id(self, token):
        return self._token2id[token]

    def to_ids(self, tokens, levels=1):
        if levels > 1:
            return [self.to_ids(item, levels - 1) for item in tokens]
        if levels == 1:
            return [self._token2id[token] for token in tokens]
        raise ValueError('Invalid number of levels')

    def from_id(self, token_id):
        return self._id2token[token_id]

    def from_ids(self, ids, levels=1):
        if levels > 1:
            return [self.from_ids(item, levels - 1) for item in ids]
        if levels == 1:
            return [self._id2token[i] for i in ids]
        raise ValueError('Invalid number of levels')
