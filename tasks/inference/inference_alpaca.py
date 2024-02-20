# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
from ascendspeed import megatron_adaptor
import modellink
from megatron.initialize import initialize_megatron
from megatron import get_args
from megatron.model import GPTModel
from megatron.arguments import core_transformer_config_from_args
from tasks.inference.infer_base import task_greedy_search, task_do_sample, task_beam_search, add_text_generate_args
from tasks.inference.infer_base import task_beam_search_with_sampling, task_return_output_log_probs, task_chat
from tasks.inference.inference_llama import model_provider


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    model = GPTModel.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    system_template = "Below is an instruction that describes a task, paired with an input that provides further " \
                      "context. Write a response that appropriately completes the request. " \
                      "Please note that you need to think through your response logically and step by step.\n\n"
    dialog_template = "### Instruction:\n{instruction}\n\n### Response:"

    task_greedy_search(args, model, system_template=system_template, dialog_template=dialog_template)
    task_do_sample(args, model, system_template=system_template, dialog_template=dialog_template)
    task_beam_search(args, model, system_template=system_template, dialog_template=dialog_template)
    task_beam_search_with_sampling(args, model, system_template=system_template, dialog_template=dialog_template)
    task_return_output_log_probs(args, model, system_template=system_template, dialog_template=dialog_template)
    task_chat(args, model, system_template=system_template, dialog_template=dialog_template)