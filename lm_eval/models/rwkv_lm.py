import os

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

os.environ["RWKV_V7_ON"] = '1'
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"


def rwkv_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    '''
    input_ids: [batch, seqlen]
    attention_mask: [batch, seqlen]
    labels: [batch, seqlen]
    '''
    batch_size, seqlen = input_ids.shape
    logits = []
    for i in range(batch_size):
        sample = input_ids[i]
        mask = attention_mask[i]
        real_length = mask.sum().item()
        sample = sample[-real_length:].tolist() # TODO: padding 0 as prefix?
        logit, _ = model(sample, None, full_output=True)
        logit = F.pad(logit, (seqlen - logit.shape[0], 0, 0, 0), "constant", 0) # TODO: maybe incorrect, but batch_size = 1 makes no sense
        logits.append(logit)
    logits = torch.stack(logits, dim=0)
    return logits

def rwkv_generate(
    pipeline,
    ctx,
    max_length: int,
    temperature: float = 0,
    top_p: float = 0.0,
    top_k: float = 1,
    **kwargs
):
    # [WARN] batch size can only be 1 for now
    # i.e. ctx.shape = (1, seqlen)
    from rwkv.utils import PIPELINE_ARGS
    args = PIPELINE_ARGS(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    # below is copied from rwkv.utils.PIPELINE.generate and it's arguments
    token_count = max_length - ctx.shape[1]
    state = None
    # [copy start]
    all_tokens = [0] + ctx[0].tolist()
    occurrence = {}
    for i in range(token_count):

        # forward & adjust prob.
        tokens = all_tokens if i == 0 else [token]
        while len(tokens) > 0:
            out, state = pipeline.model.forward(tokens[:args.chunk_len], state)
            tokens = tokens[args.chunk_len:]

        for n in args.token_ban:
            out[n] = -float('inf')
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        # sampler
        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
        if token in args.token_stop:
            break
        all_tokens += [token]
        for xxx in occurrence:
            occurrence[xxx] *= args.alpha_decay

        ttt = pipeline.decode([token])
        www = 1
        if ttt in ' \t0123456789':
            www = 0
        if token not in occurrence:
            occurrence[token] = www
        else:
            occurrence[token] += www
    # [copy end]
    res = torch.Tensor(all_tokens[1:]).unsqueeze(0)
    # print(f'[debug] res: {tokenizer.decode(res[0], skip_special_tokens=True)}')
    return res



@register_model("rwkv")
class RWKVWrapper(HFLM):

    def __init__(
        self,
        pretrained: str,    # path to weight
        tokenizer: str,     # tokenizer name (use tokenizer from flash-linear-attention)
        **kwargs
    ) -> None:
        from rwkv.utils import PIPELINE
        if "backend" in kwargs:
            assert kwargs["backend"] == "causal"
        kwargs["max_length"] = kwargs.get("max_length", 2048)
        self._config = AutoConfig.from_pretrained(tokenizer)
        super().__init__(
            pretrained=pretrained,
            backend="causal",
            tokenizer=tokenizer,
            **kwargs
        )
        self.pipeline = PIPELINE(self._model, "rwkv_vocab_v20230424")

    def _create_model(self, pretrained: str, **kwargs):
        from rwkv.model import RWKV
        model = RWKV(model=pretrained, strategy="cuda bf16")
        self._model = model

    def _get_config(self, pretrained: str, *args, **kwargs) -> None:
        pass

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ):
                return rwkv_forward(self._model, inps, attn_mask, labels)
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        return rwkv_generate(self.pipeline, context, max_length, **generation_kwargs)
