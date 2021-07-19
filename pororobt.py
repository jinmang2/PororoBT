import copy
import time

import torch
from fairseq import utils

from pororo import Pororo

from datasets import load_dataset
import pandas as pd


class PandasMixin:
    def to_csv(self):
        pass


class PororoBackTranslator(PandasMixin):
    _lang_to_code = {
        "ko": 'ko_KR',
        "en": "en_XX",
        "ja": "ja_XX",
        "zh": "zh_CN"
    }
    
    def __init__(self):
        mt = Pororo(task="machine_translation", lang="multi")
        self.hub = mt._model
        self.eng_tok = mt._tokenizer
        from fairseq import utils
        self.utils = utils
        self.max_positions = utils.resolve_max_positions(
            self.hub.task.max_positions(), 
            *[model.max_positions() for model in self.hub.models]
        )
        
    def _get_lang_code(self, lang):
        code = self._lang_to_code.get(lang, None)
        if code is None:
            raise AttributeError(f"lang must be {self._lang_to_code.keys()}")
        return code
        
    def __call__(
        self, 
        sentences, 
        init="ko", 
        mid="en",
        n_best=1,
        beam=5,
        sampling=False,
        temperature=1.0,
        sampling_topk=-1,
        sampling_topp=-1,
        max_len_a=1,
        max_len_b=50,
        no_repeat_ngram_size=4,
        lenpen=1.0,
    ):
        if init != "en":
            sentences = [
                " ".join([c if c != " " else "▁" for c in sentence])
                for sentence in sentences
            ]
        else:
            # 나중에 뽀로로보고 구현하길
            pass
            
        return self._back_translate(
            sentences, 
            init, 
            mid,
            n_best=n_best,
            beam=beam,
            sampling=sampling,
            temperature=temperature,
            sampling_topk=sampling_topk,
            sampling_topp=sampling_topp,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            no_repeat_ngram_size=no_repeat_ngram_size,
            lenpen=lenpen,
        )
        
    def _back_translate(self, sentences, init, mid, n_best, **kwargs):
        init_code = self._get_lang_code(init)
        mid_code = self._get_lang_code(mid)
        translated = self._translate(sentences, init_code, mid_code, n_best, **kwargs)
        if n_best == 1:
            translated = [s[0] for s in translated]
        back_translated = self._translate(translated, mid_code, init_code, n_best, **kwargs)
        # post processing
        results = [o[0].replace(" ", "").replace("▁", " ").strip() for o in back_translated]
        return results
        
    def _translate(self, sentences, init_code, mid_code, n_best=1, **kwargs):
        tokenized_sentences = [
            self.hub.encode(f"[{init_code}] {s} [{mid_code}]") for s in sentences
        ]
        gen_args = copy.copy(self.hub.args)
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
            generator = mt._model.task.build_generator(mt._model.models, gen_args)
            
        results = []
        total = len(sentences)
        count = 0
        init_time = time.time()
        for batch in self._build_batches(self.hub, tokenized_sentences):
            start_time = time.time()
            ids, src_tokens, src_lengths = batch
            src_tokens = src_tokens.to(self.hub.device)
            src_lengths = src_lengths.to(self.hub.device)
            sample = {
                "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}
            }
            translations = self.hub.task.inference_step(
                generator, self.hub.models, sample
            )
            for (iden, hypos) in zip(ids.tolist(), translations):
                results.append((iden, hypos))
            count += ids.shape[0]
            it_time = time.time() - start_time
            print(f"\rDone [{count:05}] / [{total}] | {it_time:.4f}/it", end="")
        print(f"\nDone {time.time() - init_time}")
        
        # sort output to match input order
        outputs = []
        for (_, hypos) in sorted(results, key=lambda x: x[0]):
            hypotheses = []
            # Process top predictions
            for hypo in hypos[: min(len(hypos), n_best)]:
                hypo_tokens = hypo["tokens"].int().cpu()
                hypotheses.append(hub.decode(hypo_tokens))
            outputs.append(hypotheses)
        return outputs
        
    def _build_batches(self, hub, tokens):
        lengths = torch.LongTensor([t.numel() for t in tokens])
        itr = hub.task.get_batch_iterator(
            dataset=hub.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=hub.args.max_tokens,
            max_sentences=hub.args.max_sentences,
            max_positions=utils.resolve_max_positions(
                hub.task.max_positions(), *[model.max_positions() for model in hub.models]
            ),
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            yield (batch["id"], batch["net_input"]["src_tokens"], batch["net_input"]["src_lengths"])
