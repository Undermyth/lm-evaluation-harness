import itertools
import logging
from typing import Generator

import datasets

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS, get_tokenizer
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack
from lm_eval.tasks.ruler.template import RWKV_TEMPLATE


TEMPLATE = """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} for {query} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?"""
eval_logger = logging.getLogger(__name__)


def download_dataset(df: Generator) -> dict[str, datasets.Dataset]:
    return {"test": datasets.Dataset.from_list(list(itertools.chain.from_iterable(df)), split=datasets.Split.TEST)}


def _prepare_niah_config(task_name: str, **kwargs) -> dict:
    """Pop the shared niah task options from ``kwargs`` and log the config.

    Popped options (all read from the task ``metadata``):
      - ``use_template``: kept for backward compatibility, currently unused.
      - ``shuffle``: shuffle the essay haystack for every sample (essay haystacks only,
        ``generate_samples`` asserts otherwise).
      - ``shuffle_once``: shuffle the essay corpus once before generating samples
        (only has an effect for essay haystacks).
      - ``enable_cache``: cache the generated samples on disk and reuse them on
        subsequent runs with the same configuration.
      - ``num_samples``: number of samples to generate (default 500).

    Any remaining keyword arguments are forwarded to :func:`get_tokenizer`.
    Returns a dict with the popped options plus the resolved ``tokenizer`` and
    ``template``.
    """
    # check for template
    use_template = kwargs.pop("use_template", False)
    tokenizer = get_tokenizer(**kwargs)
    template = TEMPLATE

    # read config from metadata and log it
    shuffle = kwargs.pop("shuffle", False)
    shuffle_once = kwargs.pop("shuffle_once", False)
    enable_cache = kwargs.pop("enable_cache", False)
    num_samples = kwargs.pop("num_samples", 500)  # Default number of samples
    config_log = {
        "shuffle_once": shuffle_once,
        "shuffle": shuffle,
        "enable_cache": enable_cache,
        "samples": num_samples,
        "template": template,
    }
    eval_logger.info(f"{task_name} Configuration: {config_log}")
    eval_logger.info(f"Generating samples for {task_name}...")

    return {
        "use_template": use_template,
        "tokenizer": tokenizer,
        "template": template,
        "shuffle": shuffle,
        "shuffle_once": shuffle_once,
        "enable_cache": enable_cache,
        "num_samples": num_samples,
    }


def niah_single_1(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_single_1", **kwargs)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="repeat"),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="repeat",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )


def niah_single_2(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_single_2", **kwargs)

    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay", shuffle=opts["shuffle_once"]),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )


def niah_single_3(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_single_3", **kwargs)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay", shuffle=opts["shuffle_once"]),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="uuids",
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )


def niah_single_word(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_single_word", **kwargs)

    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay", shuffle=opts["shuffle_once"]),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="words",
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )


def niah_multikey_1(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_multikey_1", **kwargs)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay", shuffle=opts["shuffle_once"]),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_k=4,
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )


def niah_multikey_2(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_multikey_2", **kwargs)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="needle"),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="needle",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )


def niah_multikey_3(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_multikey_3", **kwargs)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="needle"),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="needle",
            type_needle_k="uuids",
            type_needle_v="uuids",
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )


def niah_multivalue(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_multivalue", **kwargs)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay", shuffle=opts["shuffle_once"]),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_v=4,
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )


def niah_multiquery(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    opts = _prepare_niah_config("niah_multiquery", **kwargs)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay", shuffle=opts["shuffle_once"]),
            max_seq_length=seq,
            template=opts["template"],
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_q=4,
            num_samples=opts["num_samples"],
            shuffle=opts["shuffle"],
            enable_cache=opts["enable_cache"],
            TOKENIZER=opts["tokenizer"],
        )
        for seq in seq_lengths
    )
