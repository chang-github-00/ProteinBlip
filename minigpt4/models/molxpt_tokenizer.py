from transformers import BioGptTokenizer
import re
class MolxptTokenizer(BioGptTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )
    
    def _tokenize_mol(self, s):
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(s)]
        tokens_tagged = ["<m>"+t+"</w>" for t in tokens]
        return '<start-of-mol></w> ' + ' '.join(tokens_tagged) + ' <end-of-mol></w>'
        
    def _tokenize(self, text, bypass_tokenizer=False):
        """Returns a tokenized string."""
        pattern = r'(<start-of-mol>.*?<end-of-mol>)'  
        segments = re.split(pattern, text, flags=re.DOTALL)  
        splits = [segment.strip() for segment in segments if segment.strip()]
        split_tokens = []
        for t in splits:
            #print(t)
            #print(split_tokens)
            if "<start-of-mol>" in t:
                t = t.replace("<start-of-mol>", "").replace("<end-of-mol>", "")
                split_tokens.extend(self._tokenize_mol(t).split())
            else:
                if bypass_tokenizer:
                    t = t.split()
                else:
                    t = self.moses_tokenize(t, self.lang)
                for token in t:
                    if token:
                        split_tokens.extend(list(self.bpe(token).split(" ")))
        #print(split_tokens)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # remove BPE
        tokens = [t.replace(" ", "").replace("</w>", " ") for t in tokens]
        tokens = "".join(tokens).split()
        # detokenize
        text = self.moses_detokenize(tokens, self.lang)
        pattern = r'(<start-of-mol>.*?<end-of-mol>)'  
        segments = re.split(pattern, text, flags=re.DOTALL)  
        splits = [segment.strip() for segment in segments if segment.strip()]
        new_splits = []
        for s in splits:
            if "<start-of-mol>" in s:
                new_splits.append(s.replace(" ", "").replace("<m>", "").strip())
            else:
                new_splits.append(s) 
        text = " ".join(new_splits)
        return text

class MixgptTokenizer(BioGptTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )
    
    def _tokenize_mol(self, s):
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(s)]
        tokens_tagged = ["<m>"+t+"</w>" for t in tokens]
        return '[S]</w> ' + ' '.join(tokens_tagged)
    
    def _tokenize_protein(self, s):
        tokens_tagged = ["<a>"+t+"</w>" for t in s]
        return '[P]</w> ' + ' '.join(tokens_tagged)
    
    def _tokenize_antibody(self, s):
        tokens_tagged = ["<a>"+t+"</w>" for t in s]
        return '[A]</w> ' + ' '.join(tokens_tagged)
        
    def _tokenize(self, text, bypass_tokenizer=False):
        """Returns a tokenized string."""
        if text[:3] == '[S]':
            return self._tokenize_mol(text[3:]).split()
        elif text[:3] == '[P]':
            return self._tokenize_protein(text[3:]).split()
        else:
            return self._tokenize_antibody(text[3:]).split()

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # remove BPE
        text = "".join(tokens).replace(" ", "").replace("</w>", "").replace("<m>", "").replace("<a>", "")
        return text