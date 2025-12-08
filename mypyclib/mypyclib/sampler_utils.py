import sys
from collections import defaultdict
from collections.abc import Iterable
from itertools import repeat
from typing import Callable, Optional, TypeVar, cast

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.sampling_utils import (
    GREEDY,
    GenericStrategyKeyType,
    Strategy,
    UtilsSamplingParams,
)
from tensorrt_llm.sampling_params import SamplingParams

if sys.version_info[:2] >= (3, 12):
    pass
else:
    pass

T = TypeVar("T")


def resolve_sampling_strategy_impl(params: UtilsSamplingParams, *, vocab_size: int) -> Strategy:
    # The semantics are specified in the doc-string of SamplingParams

    temperature = params.temperature
    top_p = params.top_p
    top_k = params.top_k

    if SamplingParams.params_imply_greedy_decoding(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    ):
        return GREEDY

    # --- resolving default values
    # NB: not greedy, hence temperature != 0 if specified
    temperature = temperature or 1.0

    # NB: not greedy, hence top_p != 0 if specified
    top_p = top_p or 1.0
    # NB: not greedy, hence top_k != 1 if specified
    #     (0 and vocab_size are equivalent)
    top_k = top_k or vocab_size

    assert top_k > 1, "non-greedy sampling requires valid top_k"
    need_top_k = top_k < vocab_size
    assert top_p > 0, "non-greedy sampling requires valid top_p"
    need_top_p = top_p < 1

    if need_top_p:
        if need_top_k:
            return ("top_k_top_p", top_k, top_p, temperature)
        return ("top_p", top_p, temperature)
    if need_top_k:
        return ("top_k", top_k, temperature)
    return ("temperature", temperature)


# Due to tensorrt_llm::runtime::SamplingConfig using vectors, params
# in LlmRequest.sampling_params are either None or single-element lists.
# This helper method simplifies code using such params.
def _unwrap_singleton(p: Optional[list[T]]) -> Optional[T]:
    if p is None:
        return None
    (t,) = p
    return t


def _request_get_sampling_params_impl(request: LlmRequest) -> UtilsSamplingParams:
    sampling_config = request.sampling_config
    temperature = _unwrap_singleton(cast(Optional[list[float]], sampling_config.temperature))
    top_p = _unwrap_singleton(cast(Optional[list[float]], sampling_config.top_p))
    top_k = _unwrap_singleton(cast(Optional[list[int]], sampling_config.top_k))

    return UtilsSamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def _request_strategy_impl(request: LlmRequest, *, vocab_size: int) -> Strategy:
    params = _request_get_sampling_params_impl(request)
    return resolve_sampling_strategy_impl(params, vocab_size=vocab_size)


def _speculation_could_use_rejection_sampling_impl(
    request: LlmRequest, strategy: Optional[Strategy] = None
) -> bool:
    if strategy is None:
        strategy = _request_strategy_impl(
            request,
            vocab_size=2**31,  # vocab_size does not affect greediness
        )
    return get_draft_token_length(request) > 0 and strategy != GREEDY


def _group_requests_by_strategy_key_impl(
    requests: Iterable[LlmRequest],
    *,
    strategy_to_key: Callable[[Strategy, bool], GenericStrategyKeyType],
    pin_memory: bool = False,
    vocab_size: int,
) -> dict[tuple[GenericStrategyKeyType, bool], tuple[torch.Tensor, list[Strategy]]]:
    # NB: Client code relies on request indices in returned torch.Tensor being sorted.
    group_dict: dict[tuple[GenericStrategyKeyType, bool], tuple[list[int], list[Strategy]]] = (
        defaultdict(lambda: ([], []))
    )

    for req_index, req in enumerate(requests):
        strategy = _request_strategy_impl(req, vocab_size=vocab_size)
        speculation_needs_probs = (
            # NB: This criterion needs to be consistent with the gating of rejection sampling in
            #     process_draft_tokens.
            _speculation_could_use_rejection_sampling_impl(req, strategy)
        )
        strategy_key = strategy_to_key(strategy, speculation_needs_probs)
        group_dict_entry = group_dict[(strategy_key, speculation_needs_probs)]
        group_dict_entry[0].append(req_index)
        group_dict_entry[1].append(strategy)
    return {
        group_key: (
            torch.tensor(indices, pin_memory=pin_memory, dtype=torch.int32),
            strategies,
        )
        for group_key, (indices, strategies) in group_dict.items()
    }


def _apply_embedding_bias_impl(
    logits: torch.Tensor,
    requests: list[LlmRequest],
    request_steps: torch.Tensor,
) -> None:
    """Apply embedding bias (aka logit bias) to logits.

    Arguments:
      request_steps: Number of steps/tokens for each request.

    Modifies logits in-place.
    """
    # NB: Unfortunately, Torch provides no combination of torch.index_select (similar to
    #     torch.Tensor.gather -- allows one-to-many mapping) and addition, analogous to how
    #     torch.Tensor.scatter_add_ (and it's variant torch.Tensor.index_add_ -- allows
    #     many-to-one mapping) combine addition with torch.Tensor.scatter_.
    #
    #     Notwithstanding the previous point, there are two options:
    #         (i)  materialize a permuted bias tensor with repeated consecutive rows via
    #              torch.repeat_interleave and then use torch.Tensor.index_add_ (poor write
    #              locality / risk of false sharing)
    #        (ii)  materialize the correctly ordered bias tensor via torch.index_select and then
    #              perform a masked addition (poor read locality for request batches randomly
    #              mixing uniform and heterogeneous bias tensors, i.e., mixing slices with high
    #              and low reuse).
    #     Since read-caching is expected to help in typical cases, option (ii) is implemented here.

    # Track which logits require logit bias application
    logits_bias_mask = torch.zeros((logits.size(0),), dtype=torch.bool, pin_memory=True)

    _next_bias_index = 0

    def provision_bias_index() -> int:
        nonlocal _next_bias_index
        bias_index = _next_bias_index
        _next_bias_index += 1
        return bias_index

    # Indices of unique bias tensors
    #
    # NB: hash(torch.Tensor) is equivalent to id(torch.Tensor), and does not
    #     depend on tensor contents, cf. https://github.com/pytorch/pytorch/issues/2569
    bias_to_index: dict[torch.Tensor, int] = defaultdict(provision_bias_index)

    # Source indices for bias application
    bias_gather_indices: list[int] = []

    # Collect bias information
    req_bias = None
    for i, (req, steps) in enumerate(zip(requests, request_steps)):
        steps = int(steps.item())
        req_bias = req._py_embedding_bias_1d
        if req_bias is not None:
            logits_bias_mask[i : (i + steps)] = True
            req_bias_index = bias_to_index[req_bias]
            bias_gather_indices.extend(repeat(req_bias_index, steps))

    if not bias_to_index:
        return
    assert req_bias is not None  # otherwise bias_to_index is empty

    bias_gather_indices_cuda = torch.tensor(
        bias_gather_indices, pin_memory=True, dtype=torch.int32
    ).to(logits.device, non_blocking=True)
    logits_bias_mask_cuda = logits_bias_mask.to(logits.device, non_blocking=True)
    biases_tensor = torch.empty((len(bias_to_index), *req_bias.shape), pin_memory=True)
    biases_tensor = torch.stack(
        tuple(bias_to_index.keys()),
        out=biases_tensor,
    )
    biases_tensor_cuda = biases_tensor.to(logits.device, non_blocking=True)

    biases_tensor_cuda = torch.index_select(biases_tensor_cuda, 0, bias_gather_indices_cuda)
    # NB: Avoiding logits[bias_scatter_indices] += biases_tensor (and torch.Tensor.scatter_add_), because it
    #     is unclear if this allows for repeated indices, cf.
    #         https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.index_put_.html#torch-tensor-index-put
    #     and thus introduces read-after-write dependencies (including possible false
    #     sharing).
    logits[logits_bias_mask_cuda] += biases_tensor_cuda
