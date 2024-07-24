# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import logging
import numpy as np
from functools import partial
from typing import Any, Dict, List
from datasets import load_dataset, concatenate_datasets, interleave_datasets, Features

from modellink.tasks.preprocess.templates import Role
from modellink.tasks.preprocess.parser import InstructionDatasetAttr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LlamaFactoryInstructionHandler preprocess data supported format
FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

DATA_CONFIG = "dataset_info.json"


def get_dataset_list(data_args) -> List["InstructionDatasetAttr"]:
    """
    Map multiple dataset attributes to List["InstructionDatasetAttr"]
    through parameters and the data.json mapping file.
    """
    if data_args.input is not None:
        dataset_names = [ds.split("/")[-1].strip() for ds in data_args.input.split(",")]
    else:
        dataset_names = []

    try:
        with open(os.path.join(data_args.dataset_dir, DATA_CONFIG), "r") as f:
            dataset_info = json.load(f)
    except Exception as err:
        if len(dataset_names) != 0:
            raise ValueError(
                "Cannot open {} due to {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG), str(err))
            )
        dataset_info = None

    if dataset_info == None:
        raise ValueError(
                "Cannot load {}.".format(os.path.join(data_args.dataset_dir, DATA_CONFIG))
        )

    # Multiple Dataset Interleaving Probability
    if data_args.interleave_probs is not None:
        data_args.interleave_probs = [float(prob.strip()) for prob in data_args.interleave_probs.split(",")]

    dataset_list: List[InstructionDatasetAttr] = []
    for name in dataset_names:
        if name not in dataset_info:
            raise ValueError("Undefined dataset {} in {}.".format(name, DATA_CONFIG))

        if "script_url" in dataset_info[name]:
            dataset_attr = InstructionDatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
        else:
            dataset_attr = InstructionDatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        dataset_attr.set_attr("subset", dataset_info[name])
        dataset_attr.set_attr("folder", dataset_info[name])
        dataset_attr.set_attr("ranking", dataset_info[name], default=False)
        dataset_attr.set_attr("formatting", dataset_info[name], default="alpaca")

        if "columns" in dataset_info[name]:
            column_names = ["system", "images"]
            if dataset_attr.formatting == "alpaca":
                column_names.extend(["prompt", "query", "response", "history"])
            else:
                column_names.extend(["messages", "tools"])

            for column_name in column_names:
                dataset_attr.set_attr(column_name, dataset_info[name]["columns"])

        if dataset_attr.formatting == "sharegpt" and "tags" in dataset_info[name]:
            tag_names = (
                "role_tag",
                "content_tag",
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            )
            for tag in tag_names:
                dataset_attr.set_attr(tag, dataset_info[name]["tags"])

        dataset_list.append(dataset_attr)

    return dataset_list


def convert_alpaca_to_intermediate(sample: Dict[str, List[Any]], dataset_attr: "InstructionDatasetAttr", data_args: "DataArguments"):
    """
    format sample info
    {
      "instruction": "我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？",
      "input": "",
      "output": "中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。",
      "history": [
       [
        "回答的非常好",
        "感谢你的认可！还有什么需要我帮助的吗？"
       ]
      ]
     }
    ---->>>>
    {
        'prompt': [{'role': 'user', 'content': '回答的非常好'}, 
                {'role': 'assistant', 'content': '感谢你的认可！还有什么需要我帮助的吗？'}, 
                {'role': 'user', 'content': '我还想知道中国古代的五代十国时期和欧洲的中世纪有什么异同点？'}], 
        'response': [{'role': 'assistant', 'content': '中国的五代十国时期和欧洲的中世纪大体上是同时期的历史时期，但它们有许多重要的异同点。'}], 
        'system': [''], 
        'tools': ['']
    }
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    prompt = []
    
    if dataset_attr.history and isinstance(sample[dataset_attr.history], dict):
        for old_prompt, old_response in sample[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    content = []
    if dataset_attr.prompt and sample[dataset_attr.prompt]:
        content.append(sample[dataset_attr.prompt])

    if dataset_attr.query and sample[dataset_attr.query]:
        content.append(sample[dataset_attr.query])

    prompt.append({"role": Role.USER.value, "content": "\n".join(content)})

    if dataset_attr.response and isinstance(sample[dataset_attr.response], list):
        response = [
            {"role": Role.ASSISTANT.value, "content": content} for content in sample[dataset_attr.response]
        ]
    elif dataset_attr.response and isinstance(sample[dataset_attr.response], str):
        response = [{"role": Role.ASSISTANT.value, "content": sample[dataset_attr.response]}]
    else:
        response = []

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(sample[dataset_attr.system] if dataset_attr.system else "")
    outputs["tools"].append("")
    return outputs


def convert_sharegpt_to_intermediate(
    sample: Dict[str, List[Any]], dataset_attr: "InstructionDatasetAttr", data_args: "DataArguments"
):
    """
    convert sharegpt or openAI sharegpt to intermediate format
    sharegpt:
    [
    {
        "conversations": [
        {
            "from": "human",
            "value": ""
        },
        {
            "from": "function_call",
            "value": ""
        },
        {
            "from": "observation",
            "value": ""
        },
        {
            "from": "gpt",
            "value": ""
        }
        ],
        "system": "系统提示词（选填）",
        "tools": ""
    }
    ]

    ---->>>>

    {
        'prompt': [{'role': 'user', 'content': ''}, 
                {'role': 'assistant', 'content': ''}, 
                {'role': 'user', 'content': ''}], 
        'response': [{'role': 'assistant', 'content': ''}], 
        'system': [''], 
        'tools': ['']
    }
    """
    outputs = {"prompt": [], "response": [], "system": [], "tools": []}

    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,               # "human" : "user"
        dataset_attr.assistant_tag: Role.ASSISTANT.value,     # "gpt" : "assistant"
        dataset_attr.observation_tag: Role.OBSERVATION.value, # "observation" : "observation"
        dataset_attr.function_tag: Role.FUNCTION.value,       # "function_call" : "function"
        dataset_attr.system_tag: Role.SYSTEM.value,           # "system" : "system"
    }

    # "human" and "observation" must appear in odd-numbered positions
    # "gpt" and "function" must appear in even-numbered positions.
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)

    messages = sample[dataset_attr.messages]
    if dataset_attr.system_tag and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag:
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = examples[dataset_attr.system][i] if dataset_attr.system else ""

    if len(messages) == 0:
        return outputs

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning("Invalid role tag in {}.".format(messages))
            broken_data = True

        aligned_messages.append(
            {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
        )

    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
        dataset_attr.ranking and len(aligned_messages) % 2 == 0
    ):
        logger.warning("Invalid message count in {}.".format(messages))
        broken_data = True

    prompt = aligned_messages[:-1]
    response = aligned_messages[-1:]

    if broken_data:
        logger.warning("Skipping this abnormal example.")
        return outputs

    outputs["prompt"] = prompt
    outputs["response"] = response
    outputs["system"].append(system)
    outputs["tools"].append(examples[dataset_attr.tools][i] if dataset_attr.tools else "")

    return outputs


def align_dataset(dataset, dataset_attr, data_args):
    """
    Aligned dataset:
        prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        response: [{"role": "assistant", "content": "..."}]
        system: "..."
        tools: "...",
        images: []

    after doing convert_func, the features will be:
        features = Features.from_dict(
            {
                "prompt": [
                    {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
                ],
                "response": [
                    {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
                ],
                "system": [{"dtype": "string", "_type": "Value"}],
                "tools": [{"dtype": "string", "_type": "Value"}],
            }
        )
    """
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca_to_intermediate, dataset_attr=dataset_attr, data_args=data_args)
    else:
        convert_func = partial(convert_sharegpt_to_intermediate, dataset_attr=dataset_attr, data_args=data_args)
    column_names = list(next(iter(dataset)).keys())

    kwargs = dict(
        num_proc=data_args.workers,
        load_from_cache_file=(not data_args.overwrite_cache),
        desc="Converting format of dataset",
    )

    return dataset.map(
        convert_func,
        remove_columns=column_names,
        **kwargs,
    )


def merge_dataset(all_datasets, data_args):
    """Merging multiple Datasets by mix_strategy"""
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=data_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy.")


def load_single_dataset(dataset_attr, data_args):
    """loading single dataset by script/local file"""
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None

    if dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File not found.")

        if data_path is None:
            raise ValueError("File extension must be txt, csv, arrow, parquet, json or jsonl.")
    else:
        raise NotImplementedError

    kwargs = {}
    split_flag = "train"
    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        split=split_flag,
        data_files=data_files,
        cache_dir=data_args.cache_dir,
        streaming=(data_args.streaming and (dataset_attr.load_from != "file")),
        **kwargs,
    )

    if data_args.max_samples is not None:  # truncate dataset
        num_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(num_samples))

    return align_dataset(dataset, dataset_attr, data_args)