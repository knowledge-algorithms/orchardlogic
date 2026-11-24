"""Labeling client protocol and the optional OpenAI adapter.

Importing this module does not import ``openai`` and does not open a
network connection. The SDK import happens inside the live call, and only
when the caller did not inject a client.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from orchardlogic.cultivate.exceptions import LabelingError, MissingOptionalDependencyError

LABELING_EXTRA = "labeling"
DEFAULT_MAX_WORKERS = 8


@dataclass(frozen=True, slots=True)
class LabelingPrompt:
    """System instructions plus the per-node user content."""

    system: str
    user: str

    @property
    def character_count(self) -> int:
        return len(self.system) + len(self.user)


class LabelingClient(Protocol):
    """One synchronous completion. Tests pass a fake with this method."""

    def complete(self, *, model: str, prompt: LabelingPrompt) -> str:
        """Return the raw model text for one labeling prompt."""


def missing_labeling_error() -> MissingOptionalDependencyError:
    """Loud failure when a live call needs the optional OpenAI extra."""
    return MissingOptionalDependencyError(
        LABELING_EXTRA,
        "orchardlogic[labeling] is required for a live labeling call. "
        "Install that extra, or pass client=.",
    )


def _import_openai() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise missing_labeling_error() from exc
    return openai


def _complete_indexed(
    complete: Callable[..., str],
    *,
    model: str,
    prompts: Sequence[tuple[int, LabelingPrompt]],
    max_workers: int,
) -> dict[int, str]:
    """Run one completion per request index. Failed indexes are omitted."""
    if not prompts:
        return {}
    workers = max_workers if type(max_workers) is int and max_workers >= 1 else 1
    workers = min(workers, len(prompts))

    def one(prompt: LabelingPrompt) -> str:
        return complete(model=model, prompt=prompt)

    results: dict[int, str] = {}
    if workers == 1:
        for index, prompt in prompts:
            try:
                results[index] = one(prompt)
            except MissingOptionalDependencyError:
                raise
            except Exception:
                continue
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, prompt): index for index, prompt in prompts}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except MissingOptionalDependencyError:
                raise
            except Exception:
                continue
    return results


def complete_many(
    client: LabelingClient,
    *,
    model: str,
    prompts: Sequence[tuple[int, LabelingPrompt]],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[int, str]:
    """Run one wave and join successes by integer request index.

    A client that implements ``complete_many`` is used as-is, so a fake can
    stay ordered. Otherwise the prompts run on a thread pool. Indexes that
    raise are omitted; the caller retries those ids one at a time.
    """
    custom = getattr(client, "complete_many", None)
    if callable(custom):
        return custom(model=model, prompts=prompts, max_workers=max_workers)
    return _complete_indexed(
        client.complete,
        model=model,
        prompts=prompts,
        max_workers=max_workers,
    )


class OpenAILabelingClient:
    """Chat-completions adapter for labeling prompts.

    ``sdk_client`` is the injected OpenAI SDK object. When it is omitted,
    the first ``complete`` call imports ``openai`` and constructs ``OpenAI()``.
    ``complete_many`` uses a thread pool and the same request body as
    ``complete``.
    """

    def __init__(self, sdk_client: Any | None = None) -> None:
        self._sdk_client = sdk_client

    def complete(self, *, model: str, prompt: LabelingPrompt) -> str:
        sdk = self._sdk_client
        if sdk is None:
            module = _import_openai()
            sdk = module.OpenAI()
            self._sdk_client = sdk
        response = sdk.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="none",
        )
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LabelingError(
                "label card schema mismatch: empty model response"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise LabelingError("label card schema mismatch: empty model response")
        return content

    def complete_many(
        self,
        *,
        model: str,
        prompts: Sequence[tuple[int, LabelingPrompt]],
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> dict[int, str]:
        """Concurrent chat completions for one wave. No Batch API."""
        return _complete_indexed(
            self.complete,
            model=model,
            prompts=prompts,
            max_workers=max_workers,
        )
