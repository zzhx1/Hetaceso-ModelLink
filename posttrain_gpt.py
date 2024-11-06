# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from modellink import megatron_adaptor
from modellink.tasks.post_train.launcher import AutoTrainer


def launch():
    trainer = AutoTrainer()
    trainer.train()


if __name__ == '__main__':
    launch()
