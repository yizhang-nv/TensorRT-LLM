/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bindings.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tensorrt_llm/thop/attentionOp.h>
#include <tensorrt_llm/thop/moeAlltoAllMeta.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace tensorrt_llm::pybind::thop
{

void initBindings(pybind11::module_& m)
{
    // Export MoE A2A constants
    for (auto const& kv : torch_ext::moe_comm::getMoeA2AMetaInfoIndexPairs())
    {
        m.attr(kv.first) = py::int_(kv.second);
    }

    m.def("attention", &torch_ext::attention,
        // Parameters with default values using std::nullopt for optional arguments
        py::arg("q"), py::arg("k").none(true), py::arg("v").none(true), py::arg("output"),
        py::arg("output_sf").none(true), py::arg("out_dtype").none(true), py::arg("workspace_").none(true),
        py::arg("sequence_length"), py::arg("host_past_key_value_lengths"), py::arg("host_total_kv_lens"),
        py::arg("context_lengths"), py::arg("host_context_lengths"), py::arg("host_request_types"),
        py::arg("kv_cache_block_offsets").none(true), py::arg("host_kv_cache_block_offsets").none(true),
        py::arg("host_kv_cache_pool_pointers").none(true), py::arg("host_kv_cache_pool_mapping").none(true),
        py::arg("cache_indirection").none(true), py::arg("kv_scale_orig_quant").none(true),
        py::arg("kv_scale_quant_orig").none(true), py::arg("out_scale").none(true),
        py::arg("rotary_inv_freq").none(true), py::arg("rotary_cos_sin").none(true), py::arg("latent_cache").none(true),
        py::arg("q_pe").none(true), py::arg("block_ids_per_seq").none(true), py::arg("attention_sinks").none(true),
        py::arg("is_fused_qkv"), py::arg("update_kv_cache"), py::arg("predicted_tokens_per_seq"), py::arg("layer_idx"),
        py::arg("num_heads"), py::arg("num_kv_heads"), py::arg("head_size"), py::arg("tokens_per_block").none(true),
        py::arg("max_num_requests"), py::arg("max_context_length"), py::arg("attention_window_size"),
        py::arg("sink_token_length"), py::arg("beam_width"), py::arg("mask_type"), py::arg("quant_mode"),
        py::arg("q_scaling"), py::arg("position_embedding_type"), py::arg("rotary_embedding_dim"),
        py::arg("rotary_embedding_base"), py::arg("rotary_embedding_scale_type"), py::arg("rotary_embedding_scales"),
        py::arg("rotary_embedding_max_position_info"), py::arg("use_paged_context_fmha"),
        py::arg("attention_input_type").none(true), py::arg("is_mla_enable"),
        py::arg("chunked_prefill_buffer_batch_size").none(true), py::arg("q_lora_rank").none(true),
        py::arg("kv_lora_rank").none(true), py::arg("qk_nope_head_dim").none(true),
        py::arg("qk_rope_head_dim").none(true), py::arg("v_head_dim").none(true),
        py::arg("mrope_rotary_cos_sin").none(true), py::arg("mrope_position_deltas").none(true),
        py::arg("mla_tensor_params"), py::arg("attention_chunk_size").none(true),
        py::arg("softmax_stats_tensor").none(true), py::arg("spec_decoding_bool_params"),
        py::arg("spec_decoding_tensor_params"), py::arg("sparse_kv_indices").none(true),
        py::arg("sparse_kv_offsets").none(true), py::arg("sparse_attn_indices").none(true),
        py::arg("sparse_attn_offsets").none(true), py::arg("sparse_attn_indices_block_size"),
        py::arg("sparse_mla_topk").none(true), py::arg("cu_q_seqlens").none(true), py::arg("cu_kv_seqlens").none(true),
        py::arg("fmha_scheduler_counter").none(true), py::arg("mla_bmm1_scale").none(true),
        py::arg("mla_bmm2_scale").none(true), py::arg("quant_q_buffer").none(true), "Multi-head attention operation",
        py::call_guard<py::gil_scoped_release>());
}
} // namespace tensorrt_llm::pybind::thop
