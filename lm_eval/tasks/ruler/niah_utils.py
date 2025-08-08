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


def niah_single_1(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="repeat"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="repeat",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=500,
            TOKENIZER=get_tokenizer(**kwargs),
        )
        for seq in seq_lengths
    )


def niah_single_2(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)

    # check for template
    use_template = kwargs.pop("use_template", False)
    tokenizer = get_tokenizer(**kwargs)
    if use_template is True:
        if "rwkv" in tokenizer.name_or_path:
            template = RWKV_TEMPLATE.format(task_template=TEMPLATE)
        else:
            eval_logger.warning("Template is only supported for RWKV-7")
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
    eval_logger.info(f"NIAH-2 Configuration: {config_log}")
    eval_logger.info("Generating samples for niah_single_2...")

    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay", shuffle=shuffle_once),
            max_seq_length=seq,
            template=template,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=num_samples,
            shuffle=shuffle,
            enable_cache=enable_cache,
            TOKENIZER=tokenizer,
        )
        for seq in seq_lengths
    )


def niah_single_3(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="uuids",
            num_samples=500,
            TOKENIZER=get_tokenizer(**kwargs),
        )
        for seq in seq_lengths
    )


def niah_single_word(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)

    # check for template
    use_template = kwargs.pop("use_template", False)
    tokenizer = get_tokenizer(**kwargs)
    if use_template is True:
        if "rwkv" in tokenizer.name_or_path:
            template = RWKV_TEMPLATE.format(task_template=TEMPLATE)
        else:
            eval_logger.warning("Template is only supported for RWKV-7")
    template = TEMPLATE

    # read config from metadata and log it
    shuffle = kwargs.pop("shuffle", False)
    enable_cache = kwargs.pop("enable_cache", False)
    num_samples = kwargs.pop("num_samples", 500)  # Default number of samples
    config_log = {"shuffle": shuffle, "enable_cache": enable_cache, "samples": num_samples, "template": template}
    eval_logger.info(f"NIAH-Word Configuration: {config_log}")
    eval_logger.info("Generating samples for niah_single_word...")

    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=template,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="words",
            num_samples=num_samples,
            shuffle=shuffle,
            enable_cache=enable_cache,
            TOKENIZER=tokenizer,
        )
        for seq in seq_lengths
    )


def niah_multikey_1(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_k=4,
            num_samples=500,
            TOKENIZER=get_tokenizer(**kwargs),
        )
        for seq in seq_lengths
    )


def niah_multikey_2(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="needle"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="needle",
            type_needle_k="words",
            type_needle_v="numbers",
            num_samples=500,
            TOKENIZER=get_tokenizer(**kwargs),
        )
        for seq in seq_lengths
    )


def niah_multikey_3(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="needle"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="needle",
            type_needle_k="uuids",
            type_needle_v="uuids",
            num_samples=500,
            TOKENIZER=get_tokenizer(**kwargs),
        )
        for seq in seq_lengths
    )


def niah_multivalue(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_v=4,
            num_samples=500,
            TOKENIZER=get_tokenizer(**kwargs),
        )
        for seq in seq_lengths
    )


def niah_multiquery(**kwargs):
    seq_lengths = kwargs.pop("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    return download_dataset(
        generate_samples(
            get_haystack(type_haystack="essay"),
            max_seq_length=seq,
            template=TEMPLATE,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_q=4,
            num_samples=500,
            TOKENIZER=get_tokenizer(**kwargs),
        )
        for seq in seq_lengths
    )
