"""Baselines runner with spend-cap enforcement (Req 3.1-3.6, 2.2).

Three baselines are evaluated on the Test_Split, each scored by the Task_Metric
with a 95% bootstrap CI:

* base zero-shot,
* base few-shot (exemplars drawn from the Train_Split ONLY), and
* hosted few-shot (a hosted model via :class:`fts.model_client.ModelClient`).

Hosted spend is priced from token counts; the runner STOPS before a request
would exceed the Spend_Cap and signals a breach. Each baseline records its
prompt template. The exemplar-selection and spend-accounting loops are pure,
importable functions so they can be property-tested without a network.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from fts.data import Example, Splits
from fts.eval_core import BootstrapCI, bootstrap_ci


@dataclass(frozen=True)
class BaselineResult:
    """Result of one baseline run."""

    name: str
    metric: BootstrapCI
    spend_usd: float
    prompt_template: str


class SpendCapExceeded(Exception):
    """Raised when a hosted request would exceed the configured Spend_Cap."""

    def __init__(self, spend_usd: float, spend_cap_usd: float, completed: int) -> None:
        self.spend_usd = spend_usd
        self.spend_cap_usd = spend_cap_usd
        self.completed = completed
        super().__init__(
            f"spend cap ${spend_cap_usd:.6f} would be exceeded "
            f"(spent ${spend_usd:.6f} over {completed} completed requests)"
        )


@dataclass(frozen=True)
class SpendResult:
    """Outcome of a capped spend-accounting loop."""

    spend_usd: float
    completed: int
    capped: bool


def select_few_shot_exemplars(
    splits: Splits, k: int, seed: int
) -> tuple[Example, ...]:
    """Select up to ``k`` few-shot exemplars from the Train_Split ONLY.

    Selection is deterministic under ``seed`` and never draws from the
    Validation or Test splits, so returned ids are always a subset of Train and
    disjoint from Val/Test. Returns fewer than ``k`` when Train is smaller.
    """
    if k <= 0:
        return ()
    train = list(splits.train)
    # Deterministic order independent of input order: sort by a seeded key.
    import hashlib

    def key(ex: Example) -> str:
        return hashlib.sha256(f"{seed}:{ex.example_id}".encode("utf-8")).hexdigest()

    train.sort(key=key)
    return tuple(train[:k])


def accumulate_spend(
    token_counts: Sequence[tuple[int, int]],
    price_per_prompt_token: float,
    price_per_completion_token: float,
    spend_cap_usd: float,
) -> SpendResult:
    """Price requests in order, stopping BEFORE the cap is exceeded.

    ``token_counts`` is a sequence of ``(prompt_tokens, completion_tokens)`` for
    each pending request. The cost of a request is
    ``prompt_tokens * price_per_prompt_token
    + completion_tokens * price_per_completion_token``. A request is completed
    only when the running total *after* adding it would not exceed
    ``spend_cap_usd``; the first request that would breach the cap is NOT
    completed and the loop stops. The reported ``spend_usd`` equals the exact
    sum over completed requests.
    """
    spend = 0.0
    completed = 0
    for prompt_tokens, completion_tokens in token_counts:
        cost = (
            prompt_tokens * price_per_prompt_token
            + completion_tokens * price_per_completion_token
        )
        if spend + cost > spend_cap_usd:
            return SpendResult(spend_usd=spend, completed=completed, capped=True)
        spend += cost
        completed += 1
    return SpendResult(spend_usd=spend, completed=completed, capped=False)


# Documented default prompt templates (owner may swap at M0).
ZERO_SHOT_TEMPLATE = (
    "Extract the fields as a JSON object.\nInput: {inputs}\nJSON:"
)
FEW_SHOT_TEMPLATE = (
    "Extract the fields as a JSON object. Examples follow.\n"
    "{exemplars}\nInput: {inputs}\nJSON:"
)
HOSTED_FEW_SHOT_TEMPLATE = FEW_SHOT_TEMPLATE


def run_baselines(
    splits: Splits,
    few_shot_exemplars: int,
    spend_cap_usd: float,
    seed: int,
    predict: Callable[[str, Example], str],
    hosted_token_counts: Sequence[tuple[int, int]] | None = None,
    price_per_prompt_token: float = 0.0,
    price_per_completion_token: float = 0.0,
) -> list[BaselineResult]:
    """Run the three baselines on the Test_Split, each with a 95% bootstrap CI.

    ``predict(template, example)`` returns the model's prediction string for a
    Test example under the given prompt template; it is injected so the pure
    runner needs no network (production wires it to a
    :class:`fts.model_client.ModelClient`). Few-shot exemplars are drawn from
    Train only. Hosted spend is accounted from ``hosted_token_counts`` (one
    ``(prompt, completion)`` pair per hosted request) and the run raises
    :class:`SpendCapExceeded` when the cap would be breached, after recording
    the spend over completed requests.
    """
    from fts.metric import score_example

    exemplars = select_few_shot_exemplars(splits, few_shot_exemplars, seed)
    few_shot_block = "\n".join(
        f"Input: {ex.inputs}\nJSON: {ex.target}" for ex in exemplars
    )

    def eval_metric(template: str) -> BootstrapCI:
        scores = [
            score_example(predict(template, ex), ex.target) for ex in splits.test
        ]
        if not scores:
            return BootstrapCI(point=0.0, lower=0.0, upper=0.0)
        return bootstrap_ci(scores, seed=seed)

    results: list[BaselineResult] = []

    results.append(
        BaselineResult(
            name="base-zero-shot",
            metric=eval_metric(ZERO_SHOT_TEMPLATE),
            spend_usd=0.0,
            prompt_template=ZERO_SHOT_TEMPLATE,
        )
    )

    few_shot_prompt = FEW_SHOT_TEMPLATE.format(exemplars=few_shot_block, inputs="{inputs}")
    results.append(
        BaselineResult(
            name="base-few-shot",
            metric=eval_metric(few_shot_prompt),
            spend_usd=0.0,
            prompt_template=few_shot_prompt,
        )
    )

    # Hosted few-shot: enforce the spend cap over the hosted requests.
    token_counts = hosted_token_counts if hosted_token_counts is not None else []
    spend = accumulate_spend(
        token_counts,
        price_per_prompt_token,
        price_per_completion_token,
        spend_cap_usd,
    )
    hosted_prompt = HOSTED_FEW_SHOT_TEMPLATE.format(
        exemplars=few_shot_block, inputs="{inputs}"
    )
    if spend.capped:
        raise SpendCapExceeded(spend.spend_usd, spend_cap_usd, spend.completed)

    results.append(
        BaselineResult(
            name="hosted-few-shot",
            metric=eval_metric(hosted_prompt),
            spend_usd=spend.spend_usd,
            prompt_template=hosted_prompt,
        )
    )
    return results
