# coding=utf-8
# Copyright 2025 test-time-training and The HuggingFace Inc. team.
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
#
# This implementation is based on the original TTT-Linear/MLP paper:
# "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
# by Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang,
# Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto,
# and Carlos Guestrin.
# Paper: https://arxiv.org/abs/2407.04620
# Original PyTorch implementation: https://github.com/test-time-training/ttt-lm-pytorch
# Original JAX implementation: https://github.com/test-time-training/ttt-lm-jax
#
# Original code is licensed under MIT License:
# Copyright (c) 2024 test-time-training
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Transformers integration and performance optimizations:
# Copyright (c) 2025 RetentionLabs team
#
# Licensed under Apache License, Version 2.0
from torch import nn

from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from ..ttt_linear.configuration_ttt_linear import TTTLinearConfig
from ..ttt_linear.modeling_ttt_linear import (
    TTTRMSNorm,
    TTTSwiGluMLP,
    TTTRotaryEmbedding,
    TTTCausalConv1d,
    TTTMultiHeadLayerNorm,
    TTTDynamicLearningGate,
    TTTAdaptiveLinear,
    TTTLinearMemory,
    TTTLinearCache,
    TTTLinearAdaptation,
    TTTLinearLayer,
    TTTLinearPreTrainedModel,
    TTTLinearOutput,
    TTTLinearCausalLMOutput,
    TTTLinearModel,
    TTTLinearForCausalLM,
    TTTLinearForImageClassification
)

from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.import_utils import is_causal_conv1d_available
if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None


logger = logging.get_logger(__name__)


class TTTMLPConfig(TTTLinearConfig):
    pass


class TTTMLPMemory(TTTLinearMemory):
    depth = 2

    @property
    def struct_detail(self):
        return [
            TTTAdaptiveLinear(self.num_heads, self.head_dim, 4 * self.head_dim),
            TTTAdaptiveLinear(self.num_heads, 4 * self.head_dim, self.head_dim)
        ]


class TTTMLPCache(TTTLinearCache):
    pass


class TTTMLPAdaptation(TTTLinearAdaptation):
    memory_class = TTTMLPMemory


class TTTMLPLayer(TTTLinearLayer):
    def __init__(self, config: TTTMLPConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.pre_conv = config.pre_conv

        self.self_adapt = TTTMLPAdaptation(config=config, layer_idx=layer_idx)

        self.mlp = TTTSwiGluMLP(config)
        if self.pre_conv:
            self.conv = TTTCausalConv1d(config, layer_idx)

        self.seq_norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx


class TTTMLPPreTrainedModel(TTTLinearPreTrainedModel):
    config_class = TTTMLPConfig
    base_model_prefix = "ttt_mlp"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TTTMLPLayer"]


class TTTMLPOutput(TTTLinearOutput):
    pass


class TTTMLPCausalLMOutput(TTTLinearCausalLMOutput):
    pass


class TTTMLPModel(TTTLinearModel):
    def __init__(self, config: TTTMLPConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TTTMLPLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = TTTRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TTTRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class TTTMLPForCausalLM(TTTLinearForCausalLM):
    def __init__(self, config: TTTMLPConfig):
        super().__init__(config)
        self.model = TTTMLPModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class TTTMLPForSequenceClassification(GenericForSequenceClassification, TTTMLPPreTrainedModel):
    pass


class TTTMLPForTokenClassification(GenericForTokenClassification, TTTMLPPreTrainedModel):
    pass


class TTTMLPForImageClassification(TTTLinearForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.vit = TTTMLPModel(config)


__all__ = [
    "TTTMLPConfig",
    "TTTMLPMemory",
    "TTTMLPAdaptation",
    "TTTMLPLayer",
    "TTTMLPPreTrainedModel",
    "TTTMLPModel",
    "TTTMLPForCausalLM",
    "TTTMLPForSequenceClassification",
    "TTTMLPForTokenClassification",
    "TTTMLPForImageClassification"
]
