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

import abc
import sys
import types
import argparse
import torch


class MegatronAdaptation:
    """
        A module manager supports adaptation registration, application and execution.
    """
    _patch_info_collection = {}
    _args = None

    @classmethod
    def execute(cls):
        """
        Execute adaptations.
        """
        MegatronAdaptation.pre_execute()
        for adaptation in [CoreAdaptation(), LegacyAdaptation()]:
            adaptation.execute()
        MegatronAdaptation.apply()
        MegatronAdaptation.post_execute()

    @classmethod
    def register(cls, orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        """
        Register adaptations into collection.
        """
        if orig_func_name not in cls._patch_info_collection:
            from mindspeed.patch_utils import Patch
            cls._patch_info_collection[orig_func_name] = Patch(orig_func_name, new_func, create_dummy)
        else:
            cls._patch_info_collection.get(orig_func_name).set_patch_func(new_func, force_patch)

    @classmethod
    def apply(cls):
        """
        Apply adaptations.
        """
        for patch in cls._patch_info_collection.values():
            patch.apply_patch()

    @classmethod
    def get_args(cls):
        if cls._args is not None:
            return cls._args

        from modellink.training.arguments import process_args
        parser = argparse.ArgumentParser(description='ModelLink Arguments', allow_abbrev=False)
        _args, _ = process_args(parser).parse_known_args()
        return _args

    @classmethod
    def pre_execute(cls):
        """
        Execute before other adaptations.
        """
        def _process_args_dummy(parser):
            parser.conflict_handler = 'resolve'
            group = parser.add_argument_group(title='dummy')
            group.add_argument('--optimization-level', type=int, default=-1)
            return parser

        def _by_pass_ms_core():
            MegatronAdaptation.register('mindspeed.arguments.process_args', _process_args_dummy)
            MegatronAdaptation.apply()
            sys.modules['transformer_engine'] = types.ModuleType('dummy')

        # ms-core dependencies should be import after _by_pass_ms_core
        _by_pass_ms_core()

        from mindspeed.patch_utils import MindSpeedPatchesManager
        from mindspeed.megatron_adaptor import te_adaptation, apex_adaptation, torch_adaptation
        te_adaptation(MindSpeedPatchesManager)
        apex_adaptation(MindSpeedPatchesManager)
        torch_adaptation(MindSpeedPatchesManager)
        MindSpeedPatchesManager.apply_patches()

    @classmethod
    def post_execute(cls):
        """
        Execute after other adaptations.
        """
        from ..core import build_layers_wrapper
        from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.transformer_block import TransformerBlock
        # For MOE + Ascend MC2, here we can only execute this after _transformer_block_build_layers takes effect.
        TransformerBlock._build_layers = build_layers_wrapper(TransformerBlock._build_layers,
                                                              ColumnParallelLinear.forward,
                                                              RowParallelLinear.forward)


class MegatronAdaptationABC:
    """
    Abstract class for adaptation.
    """
    @abc.abstractmethod
    def execute(self):
        """
        Do Adaptation
        """


class CoreAdaptation(MegatronAdaptationABC):
    """
    Adaptations for models in Megatron-LM Core structure.
    """
    def execute(self):
        self.patch_fusions()
        self.patch_core_models()
        self.patch_core_transformers()
        self.patch_pipeline_parallel()
        self.patch_tensor_parallel()
        self.patch_parallel_state()
        self.patch_datasets()
        self.patch_utils()

    def patch_fusions(self):
        from mindspeed.core.fusions.fused_layer_norm import (FusedLayerNormAffineFunction, FastLayerNormFN)
        from mindspeed.core.fusions.fused_softmax import (is_kernel_available, ScaledUpperTriangMaskedSoftmax,
                                                          ScaledMaskedSoftmax, ScaledSoftmax, forward_fused_softmax)
        from mindspeed.core.fusions.fused_bias_swiglu import SwiGLUFunction, BiasSwiGLUFunction

        # use torch-npu fused layer norm
        MegatronAdaptation.register('megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction',
                                    FusedLayerNormAffineFunction)
        # use torch-npu fused layer norm
        MegatronAdaptation.register('megatron.core.fusions.fused_layer_norm.FastLayerNormFN', FastLayerNormFN)
        # use torch-npu npu_scaled_masked_softmax
        MegatronAdaptation.register('megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                                    ScaledUpperTriangMaskedSoftmax)
        MegatronAdaptation.register('megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax',
                                    ScaledMaskedSoftmax)  # use torch-npu npu_scaled_masked_softmax
        MegatronAdaptation.register('megatron.core.fusions.fused_softmax.ScaledSoftmax',
                                    ScaledSoftmax)  # use torch-npu npu_scaled_masked_softmax
        MegatronAdaptation.register('megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                                    is_kernel_available)  # replace kernel check
        MegatronAdaptation.register(
            'megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
            forward_fused_softmax)
        MegatronAdaptation.register('megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction', SwiGLUFunction)
        MegatronAdaptation.register('megatron.core.fusions.fused_bias_swiglu.BiasSwiGLUFunction',
                                    BiasSwiGLUFunction)

    def patch_core_models(self):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from mindspeed.core.models.common.embeddings.rotary_pos_embedding import get_pos_emb_on_this_cp_rank
        from ..training.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank, get_device_wrapper
        from ..core import rotary_embedding_forward, apply_rotary_pos_emb_bshd
        from ..core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec_wrapper
        from ..core.transformer.dot_product_attention import dot_product_attention_init_wrapper, \
            dot_product_attention_forward_wrapper, ulysses_context_parallel_forward_wrapper
        from ..core.transformer.attention import attention_init_wrapper
        from ..core.models.gpt.gpt_model import gpt_model_init_wrapper
        from ..core import rotary_embedding_init_wrapper, gpt_model_forward

        # Embedding
        MegatronAdaptation.register(
            'megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
            get_pos_emb_on_this_cp_rank)
        # rotary support for Megatron-LM core 0.6.0
        MegatronAdaptation.register(
            'megatron.core.models.common.embeddings.rotary_pos_embedding.apply_rotary_pos_emb_bshd',
            apply_rotary_pos_emb_bshd)
        MegatronAdaptation.register(
            'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward',
            rotary_embedding_forward)
        MegatronAdaptation.register(
            'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__',
            rotary_embedding_init_wrapper)

        # Attention
        MegatronAdaptation.register('megatron.core.transformer.attention.Attention.__init__',
                                    attention_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
                                    dot_product_attention_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                                    dot_product_attention_forward_wrapper)
        MegatronAdaptation.register(
            'megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention.__init__',
            dot_product_attention_init_wrapper)
        MegatronAdaptation.register(
            'megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention.forward',
            dot_product_attention_forward_wrapper)
        # For GQA in ulysses and hybrid
        MegatronAdaptation.register(
            'mindspeed.core.context_parallel.ulysses_context_parallel.UlyssesContextAttention.forward',
            ulysses_context_parallel_forward_wrapper)

        # Layer Definition
        # For NPU, we use local-mcore-structrue in te layer.
        MegatronAdaptation.register(
            'megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec',
            get_gpt_layer_local_spec)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec',
                                    get_gpt_layer_local_spec_wrapper)

        MegatronAdaptation.register('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
        MegatronAdaptation.register('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)
        MegatronAdaptation.register('megatron.training.dist_signal_handler.get_device', get_device_wrapper)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.forward', gpt_model_forward)
        MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.__init__', gpt_model_init_wrapper)

        # For recomputation
        from mindspeed.core.transformer.transformer_block import transformer_block_checkpointed_forward_wrapper
        MegatronAdaptation.register(
            'megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
            transformer_block_checkpointed_forward_wrapper)

    def patch_core_transformers(self):
        from mindspeed.core.transformer.moe.token_dispatcher import allgather_token_permutation, \
            allgather_token_unpermutation
        from mindspeed.core.transformer.moe.grouped_gemm_util import Ops, grouped_gemm_is_available, \
            get_device_capability
        from mindspeed.core.transformer.transformer import core_mlp_forward_wrapper

        from ..core.transformer.moe.moe_layer import moe_layer_init_wrapper, moe_layer_forward
        from ..core.transformer.transformer_block import _transformer_block_build_layers
        from ..core.transformer.transformer_layer import transformer_layer_init_wrapper
        from ..core import (PTNorm, topk_router_forward, topk_router_routing, z_loss_func,
                            TransformerLayerSubmodules,
                            transformer_layer_forward, get_num_layers_to_build_wrapper,
                            transformer_block_init_wrapper, transformer_block_forward, core_mlp_init)
        MegatronAdaptation.register('torch.cuda.get_device_capability', get_device_capability)
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TENorm', PTNorm)
        MegatronAdaptation.register('megatron.core.transformer.moe.router.TopKRouter.routing', topk_router_routing)
        MegatronAdaptation.register('megatron.core.transformer.moe.router.TopKRouter.forward', topk_router_forward)
        MegatronAdaptation.register('megatron.core.transformer.transformer_layer.TransformerLayerSubmodules',
                                    TransformerLayerSubmodules)
        MegatronAdaptation.register('megatron.core.transformer.transformer_layer.TransformerLayer.__init__',
                                    transformer_layer_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.transformer_layer.TransformerLayer.forward',
                                    transformer_layer_forward)
        MegatronAdaptation.register('megatron.core.transformer.moe.router.z_loss_func', z_loss_func)
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.get_num_layers_to_build',
                                    get_num_layers_to_build_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)
        MegatronAdaptation.register('megatron.core.transformer.moe.grouped_gemm_util.grouped_gemm_is_available',
                                    grouped_gemm_is_available)

        # Transformer block
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.__init__',
                                    transformer_block_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                    transformer_block_forward)
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.TransformerBlock._build_layers',
                                    _transformer_block_build_layers)
        MegatronAdaptation.register('megatron.core.transformer.mlp.MLP.__init__', core_mlp_init)
        MegatronAdaptation.register('megatron.core.transformer.mlp.MLP.forward', core_mlp_forward_wrapper)

        # For mcore moe
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
                                    moe_layer_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.forward', moe_layer_forward)

        args = MegatronAdaptation.get_args()
        if args.moe_permutation_async_comm and args.moe_token_dispatcher_type == 'allgather':
            from mindspeed.core.transformer.moe.router import aux_loss_load_balancing
            MegatronAdaptation.register(
                'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_permutation',
                allgather_token_permutation)
            MegatronAdaptation.register(
                'megatron.core.transformer.moe.token_dispatcher.MoEAllGatherTokenDispatcher.token_unpermutation',
                allgather_token_unpermutation)
            MegatronAdaptation.register('megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing',
                                        aux_loss_load_balancing)

        if hasattr(args, 'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute:
            from mindspeed.core.fusions.npu_moe_token_permute import permute_wrapper
            from mindspeed.core.fusions.npu_moe_token_unpermute import unpermute_wrapper
            MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.permute', permute_wrapper)
            MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.unpermute', unpermute_wrapper)

        # For drop and pad feature in all2all dispatcher
        if args.moe_expert_capacity_factor:
            from ..core.transformer.moe.router import aux_loss_load_balancing, apply_load_balancing_loss
            from ..core.transformer.moe.moe_utils import topk_softmax_with_capacity
            # balancing strategy relies on moe_expert_capacity_factor
            MegatronAdaptation.register('megatron.core.transformer.moe.router.TopKRouter.aux_loss_load_balancing',
                                        aux_loss_load_balancing)
            MegatronAdaptation.register('megatron.core.transformer.moe.router.TopKRouter.apply_load_balancing_loss',
                                        apply_load_balancing_loss)
            MegatronAdaptation.register('megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity',
                                        topk_softmax_with_capacity)

            if args.moe_token_dispatcher_type == 'alltoall':
                from ..core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
                MegatronAdaptation.register(
                    'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher',
                    MoEAlltoAllTokenDispatcher)

        # For groupMLP especially deepseek
        from mindspeed.core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward_wrapper
        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.__init__',
                                    groupedmlp_init_wrapper)
        MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward',
                                    groupedmlp_forward_wrapper)

    def patch_pipeline_parallel(self):
        from ..core.pipeline_parallel.p2p_communication import _batched_p2p_ops

        # solve send recv bug
        MegatronAdaptation.register('megatron.core.pipeline_parallel.p2p_communication._batched_p2p_ops',
                                    _batched_p2p_ops)

        # dpo relative, we need to change the recv/send shape when using PP, then deal with it by ourselves.
        from modellink.tasks.post_train.utils import get_tensor_shapes_decorator
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.get_tensor_shapes',
                                    get_tensor_shapes_decorator)

        # For recompute-in-advance
        from ..core.pipeline_parallel.schedules import get_forward_backward_func_wrapper
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.get_forward_backward_func', get_forward_backward_func_wrapper)

    def patch_tensor_parallel(self):
        from mindspeed.core.tensor_parallel.layers import vocab_parallel_embedding_forward
        from mindspeed.core.tensor_parallel.random import _set_cuda_rng_state
        from mindspeed.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy_forward
        from ..core import vocab_embedding_forward_wrapper, vocab_embedding_init_wrapper, checkpoint_forward_wrapper, checkpoint_backward_wrapper

        # default_generators need replace after set_device
        MegatronAdaptation.register('megatron.core.tensor_parallel.random._set_cuda_rng_state', _set_cuda_rng_state)
        # change masked_target for better performance
        MegatronAdaptation.register(
            'megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward',
            vocab_parallel_cross_entropy_forward)
        MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                    vocab_parallel_embedding_forward)
        MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward',
                                    vocab_embedding_forward_wrapper)
        MegatronAdaptation.register('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                                    vocab_embedding_init_wrapper)
        MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.forward',
                                    checkpoint_forward_wrapper)
        MegatronAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.backward',
                                    checkpoint_backward_wrapper)
        # For recompute-in-advance
        from mindspeed.core.tensor_parallel.random import checkpoint_wrapper
        MegatronAdaptation.register('megatron.core.tensor_parallel.random.checkpoint', checkpoint_wrapper)

    def patch_parallel_state(self):
        import megatron
        from mindspeed.core.parallel_state import (initialize_model_parallel, destroy_model_parallel_wrapper, \
                                                   get_context_parallel_group_for_send_recv_overlap)
        from ..core import initialize_model_parallel_decorator
        from ..core import destroy_model_parallel_decorator
        from ..core.transformer.transformer_block import get_layer_offset_wrapper

        # Bugfix for Megatron-LM core 0.6.0, to be removed for next version.
        MegatronAdaptation.register('megatron.core.parallel_state.initialize_model_parallel',
                                    initialize_model_parallel)
        MegatronAdaptation.register('megatron.core.parallel_state.initialize_model_parallel',
                                    initialize_model_parallel_decorator)

        # For MoE
        MegatronAdaptation.register('megatron.core.parallel_state.destroy_model_parallel',
                                    destroy_model_parallel_decorator)

        # For cp parallel state destroy
        MegatronAdaptation.register('megatron.core.parallel_state.destroy_model_parallel',
                                    destroy_model_parallel_wrapper)
        MegatronAdaptation.register('megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                                    get_context_parallel_group_for_send_recv_overlap)
        MegatronAdaptation.register('megatron.core.mpu', megatron.core.parallel_state)
        MegatronAdaptation.register(
            'megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset',
            get_layer_offset_wrapper)

    def patch_datasets(self):
        from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
        from megatron.core.datasets.gpt_dataset import GPTDataset
        from ..core import (build_generic_dataset, _build_document_sample_shuffle_indices,
                            indexed_dataset_builder_init_wrapper, add_item_wrapper, finalize_wrapper)

        # change attributions
        GPTDataset._build_document_sample_shuffle_indices = _build_document_sample_shuffle_indices
        BlendedMegatronDatasetBuilder.build_generic_dataset = build_generic_dataset
        MegatronAdaptation.register('megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.__init__',
                                    indexed_dataset_builder_init_wrapper)
        MegatronAdaptation.register('megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.add_item',
                                    add_item_wrapper)
        MegatronAdaptation.register('megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.finalize',
                                    finalize_wrapper)

    def patch_utils(self):
        from modellink.training.utils import unwrap_model_wrapper
        MegatronAdaptation.register('megatron.training.checkpointing.unwrap_model', unwrap_model_wrapper)
        MegatronAdaptation.register('megatron.training.training.unwrap_model', unwrap_model_wrapper)

        from modellink.training.utils import generate_adaptive_cp_mask_list_by_user, generate_adaptive_cp_grid_mask_by_user
        MegatronAdaptation.register('mindspeed.core.context_parallel.utils.generate_adaptive_cp_mask_list_by_user',
                                generate_adaptive_cp_mask_list_by_user)
        MegatronAdaptation.register('mindspeed.core.context_parallel.utils.generate_adaptive_cp_grid_mask_by_user',
                                generate_adaptive_cp_grid_mask_by_user)


class LegacyAdaptation(MegatronAdaptationABC):
    """
        Adaptations for models in legacy structure.
    """

    def execute(self):
        self.patch_miscellaneous()
        self.patch_model()
        self.patch_initialize()
        self.patch_training()
        self.patch_inference()
        self.patch_log_handler()
        self.patch_high_availability_feature()
        self.patch_optimizer()

    def patch_log_handler(self):
        from megatron.training.log_handler import CustomHandler
        from modellink.training.utils import emit
        CustomHandler.emit = emit

    def patch_high_availability_feature(self):
        args = MegatronAdaptation.get_args()
        from ..training import setup_model_and_optimizer_wrapper
        from ..core import (get_megatron_optimizer_wrapper, clip_grad_norm_fp32_wrapper,
                            distributed_optimizer_init_wrapper,
                            start_grad_sync_wrapper, distributed_data_parallel_init_wrapper,
                            distributed_optimizer_init_for_reuse_fp32_wrapper,
                            get_parameter_state_dp_zero_with_high_availability_wrapper)

        if args.enable_high_availability:  # already check enable_high_availability inside
            MegatronAdaptation.register(
                'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                distributed_data_parallel_init_wrapper)
            MegatronAdaptation.register('megatron.core.distributed.param_and_grad_buffer.Bucket.start_grad_sync',
                                        start_grad_sync_wrapper)
            MegatronAdaptation.register('megatron.training.training.get_megatron_optimizer',
                                        get_megatron_optimizer_wrapper)
            MegatronAdaptation.register('megatron.core.optimizer.optimizer.clip_grad_norm_fp32',
                                        clip_grad_norm_fp32_wrapper)
            MegatronAdaptation.register('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                        distributed_optimizer_init_wrapper)
            MegatronAdaptation.register('megatron.training.training.setup_model_and_optimizer',
                                        setup_model_and_optimizer_wrapper)
            if args.reuse_fp32_param:
                from mindspeed.optimizer.optimizer import mixed_precision_optimizer_step, reuse_fp32_param_init_wrapper, \
                    optimizer_config_init_wrapper
                MegatronAdaptation.register('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
                                            mixed_precision_optimizer_step)
                MegatronAdaptation.register('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                                            reuse_fp32_param_init_wrapper)
                MegatronAdaptation.register('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                                            optimizer_config_init_wrapper)

                MegatronAdaptation.register('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                            distributed_optimizer_init_for_reuse_fp32_wrapper)
                MegatronAdaptation.register('mindio_ttp.adaptor.TTPReplicaOptimizer.get_parameter_state_dp_zero_for_ttp',
                                            get_parameter_state_dp_zero_with_high_availability_wrapper)

    def patch_model(self):
        from mindspeed.core.fusions.fused_layer_norm import (FusedLayerNormAffineFunction, FastLayerNormFN)
        from mindspeed.core.fusions.fused_softmax import (is_kernel_available, ScaledUpperTriangMaskedSoftmax,
                                                          ScaledMaskedSoftmax, ScaledSoftmax, forward_fused_softmax)
        from mindspeed.core.fusions.fused_layer_norm import fused_layer_norm_affine

        from ..legacy.model.transformer import parallel_transformer_layer_init_wrapper
        from ..legacy.model.transformer import parallel_mlp_forward_wrapper
        from ..legacy.model import (
            GPTModel, parallel_transformer_init, transformer_language_model_forward_wrapper,
            state_dict_for_save_checkpoint_wrapper,
            core_attention_wrapper, core_attention_forward, FlashSelfAttention,
            ParallelAttention_wrapper, ParallelAttentionForward,
            parallel_transformer_forward, parallel_mlp_init_wrapper,
            rms_norm_init_wrapper, rms_norm_forward, post_language_model_processing
        )
        from ..training.checkpointing import load_args_from_checkpoint_wrapper

        # patch_fused_layer_norm
        MegatronAdaptation.register('megatron.legacy.model.fused_layer_norm.FusedLayerNormAffineFunction',
                                    FusedLayerNormAffineFunction)  # use torch-npu fused layer norm
        MegatronAdaptation.register('megatron.legacy.model.fused_layer_norm.FastLayerNormFN',
                                    FastLayerNormFN)  # use torch-npu fused layer norm
        MegatronAdaptation.register('megatron.legacy.model.fused_layer_norm.fused_layer_norm_affine',
                                    fused_layer_norm_affine)  # use torch-npu fused layer norm

        # patch_fused_softmax
        MegatronAdaptation.register('megatron.legacy.model.fused_softmax.ScaledUpperTriangMaskedSoftmax',
                                    ScaledUpperTriangMaskedSoftmax)  # use torch-npu npu_scaled_masked_softmax
        MegatronAdaptation.register('megatron.legacy.model.fused_softmax.ScaledMaskedSoftmax',
                                    ScaledMaskedSoftmax)  # use torch-npu npu_scaled_masked_softmax
        MegatronAdaptation.register('megatron.legacy.model.fused_softmax.ScaledSoftmax',
                                    ScaledSoftmax)  # use torch-npu npu_scaled_masked_softmax
        MegatronAdaptation.register('megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available',
                                    is_kernel_available)  # replace kernel check
        MegatronAdaptation.register(
            'megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax',
            forward_fused_softmax)

        # patch_rms_norm
        MegatronAdaptation.register('megatron.legacy.model.rms_norm.RMSNorm.__init__', rms_norm_init_wrapper)
        MegatronAdaptation.register('megatron.legacy.model.rms_norm.RMSNorm.forward', rms_norm_forward)

        # patch_transformer
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelMLP.__init__',
                                    parallel_mlp_init_wrapper)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelMLP.forward',
                                    parallel_mlp_forward_wrapper)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelTransformerLayer.__init__',
                                    parallel_transformer_layer_init_wrapper)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelTransformer.__init__',
                                    parallel_transformer_init)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelTransformer.forward',
                                    parallel_transformer_forward)
        MegatronAdaptation.register(
            'megatron.legacy.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint',
            state_dict_for_save_checkpoint_wrapper)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelAttention.__init__',
                                    ParallelAttention_wrapper)
        MegatronAdaptation.register('megatron.legacy.model.transformer.ParallelAttention.forward',
                                    ParallelAttentionForward)
        MegatronAdaptation.register('megatron.legacy.model.transformer.CoreAttention.__init__',
                                    core_attention_wrapper)
        MegatronAdaptation.register('megatron.legacy.model.transformer.CoreAttention.forward',
                                    core_attention_forward)
        MegatronAdaptation.register('megatron.legacy.model.transformer.FlashSelfAttention', FlashSelfAttention)

        # patch gptmodel
        MegatronAdaptation.register('megatron.legacy.model.GPTModel', GPTModel)
        MegatronAdaptation.register('megatron.legacy.model.gpt_model.post_language_model_processing',
                                    post_language_model_processing)
        # patch language model
        MegatronAdaptation.register('megatron.legacy.model.language_model.TransformerLanguageModel.forward',
                                    transformer_language_model_forward_wrapper)
        MegatronAdaptation.register('megatron.training.checkpointing.load_args_from_checkpoint',
                                    load_args_from_checkpoint_wrapper)

    def patch_initialize(self):
        from mindspeed.initialize import _compile_dependencies
        from ..training.initialize import initialize_megatron

        MegatronAdaptation.register('megatron.training.initialize._compile_dependencies',
                                    _compile_dependencies)  # remove cuda kernel compile
        MegatronAdaptation.register('megatron.training.initialize.initialize_megatron', initialize_megatron)

    def patch_training(self):
        from ..training import get_model_wrapper, train
        from ..training.checkpointing import load_checkpoint_wrapper
        from ..legacy.data import build_pretraining_data_loader

        MegatronAdaptation.register('megatron.training.training.get_model', get_model_wrapper)
        MegatronAdaptation.register('megatron.training.training.build_pretraining_data_loader',
                                    build_pretraining_data_loader)
        MegatronAdaptation.register('megatron.training.training.train', train)
        MegatronAdaptation.register('megatron.training.training.load_checkpoint', load_checkpoint_wrapper)

    def patch_inference(self):
        from ..inference.text_generation.tokenization import tokenize_prompts, _tokenize_prompts_and_batch
        from ..inference.text_generation.forward_step import inference_forward_step_init_wrapper, _no_pipelining_forward_step, _with_pipelining_forward_step
        from ..inference.text_generation.generation import generate_tokens_probs_and_return_on_first_stage, beam_search_and_return_on_first_stage
        
        MegatronAdaptation.register('megatron.inference.text_generation.tokenization.tokenize_prompts', tokenize_prompts)
        MegatronAdaptation.register('megatron.inference.text_generation.tokenization._tokenize_prompts_and_batch', _tokenize_prompts_and_batch)
        MegatronAdaptation.register('megatron.inference.text_generation.generation.generate_tokens_probs_and_return_on_first_stage', generate_tokens_probs_and_return_on_first_stage)
        MegatronAdaptation.register('megatron.inference.text_generation.generation.beam_search_and_return_on_first_stage', beam_search_and_return_on_first_stage)
        MegatronAdaptation.register('megatron.inference.text_generation.forward_step.ForwardStep.__init__', inference_forward_step_init_wrapper)
        MegatronAdaptation.register('megatron.inference.text_generation.forward_step._no_pipelining_forward_step', _no_pipelining_forward_step)
        MegatronAdaptation.register('megatron.inference.text_generation.forward_step._with_pipelining_forward_step', _with_pipelining_forward_step)

    def patch_miscellaneous(self):
        from modellink.training.utils import print_args_wrapper
        from modellink.training.arguments import validate_args_decorator
        from modellink.training.arguments import core_transformer_config_from_args_wrapper
        from ..training.checkpointing import _load_base_checkpoint_wrapper
        from ..training.tokenizer import build_tokenizer
        from ..training.arguments import parse_args_decorator

        MegatronAdaptation.register('megatron.training.arguments.parse_args', parse_args_decorator)
        MegatronAdaptation.register('megatron.training.arguments.validate_args', validate_args_decorator)
        # After validating arguments, do arguments printing.
        MegatronAdaptation.register('megatron.training.arguments._print_args', print_args_wrapper)
        MegatronAdaptation.register('megatron.training.global_vars.build_tokenizer', build_tokenizer)
        MegatronAdaptation.register('megatron.training.checkpointing._load_base_checkpoint',
                                    _load_base_checkpoint_wrapper)

        # For transformer layer configuration
        MegatronAdaptation.register('megatron.training.arguments.core_transformer_config_from_args',
                                    core_transformer_config_from_args_wrapper)

        # For torch 2.2.0
        torch.compile = torch.jit.script

    def patch_optimizer(self):
        args = MegatronAdaptation.get_args()
        if args.reuse_fp32_param and not args.enable_high_availability:
            from mindspeed.optimizer.optimizer import mixed_precision_optimizer_step, reuse_fp32_param_init_wrapper, \
                optimizer_config_init_wrapper
            from mindspeed.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
            MegatronAdaptation.register('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
                                        mixed_precision_optimizer_step)
            MegatronAdaptation.register(
                'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                reuse_fp32_param_init_wrapper)
            MegatronAdaptation.register('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                                        optimizer_config_init_wrapper)
            MegatronAdaptation.register('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                        reuse_fp32_param_distrib_optimizer_init_wrapper)

        if args.swap_attention:
            from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import \
                allowed_recomputing_module_wrapper
            from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import \
                setup_model_and_optimizer_wrapper
            from megatron.legacy.model.transformer import ParallelTransformerLayer
            from megatron.core.transformer.transformer_layer import TransformerLayer
            if hasattr(args, "use_mcore_models") and args.use_mcore_models:
                allowed_recomputing_module_wrapper(TransformerLayer)
            else:
                allowed_recomputing_module_wrapper(ParallelTransformerLayer)
            MegatronAdaptation.register('megatron.training.training.setup_model_and_optimizer',
                                        setup_model_and_optimizer_wrapper)


MegatronAdaptation.execute()
