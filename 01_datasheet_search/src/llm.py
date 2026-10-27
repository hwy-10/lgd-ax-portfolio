"""답변 모델 호출. 제공자가 달라도 같은 형태로 결과를 돌려준다.

모델 지정 형식   "제공자:모델ID[@effort]"
  openai:gpt-4.1-mini
  anthropic:claude-fable-5-1@low
  ollama:gemma3:4b                    이 PC 에서 도는 로컬 모델 (비용 없음)
  ollama:nemotron-3-nano:4b@nothink   사고 모드를 끈 로컬 모델

같은 평가셋에 모델만 바꿔 올리려고 만든 층이다. 검색과 판정 기준은 그대로 두고
정답률, 거절, 토큰, 지연만 비교한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from config import ANTHROPIC_API_KEY, MAX_OUTPUT_TOKENS, OLLAMA_URL, OPENAI_API_KEY

# Fable 5.1 은 사고(thinking)가 항상 켜져 있고 출력 토큰에 포함된다.
# 답변 자체는 3문장이지만 사고 몫을 남겨 둔다.
ANTHROPIC_MAX_TOKENS = 16000


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    effort: str | None = None

    @property
    def label(self) -> str:
        return f"{self.model}" + (f" ({self.effort})" if self.effort else "")


@dataclass(frozen=True)
class Completion:
    text: str
    model: str            # 실제로 답한 모델 (폴백이 일어나면 요청한 모델과 다르다)
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str = ""
    fallback_used: bool = False


def parse_spec(spec: str) -> ModelSpec:
    if ":" not in spec:
        raise ValueError(f"모델 지정은 '제공자:모델ID' 형식이어야 합니다: {spec!r}")
    provider, rest = spec.split(":", 1)
    model, _, effort = rest.partition("@")
    provider = provider.strip().lower()
    if provider not in {"openai", "anthropic", "ollama"}:
        raise ValueError(f"지원하지 않는 제공자: {provider}")
    return ModelSpec(provider=provider, model=model.strip(), effort=effort.strip() or None)


def _ollama_running() -> bool:
    import httpx

    try:
        return httpx.get(f"{OLLAMA_URL}/api/version", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def has_key(spec: str | ModelSpec) -> bool:
    """호출할 수 있는가. 클라우드는 API 키, 로컬은 Ollama 서버가 떠 있는지를 본다."""
    spec = parse_spec(spec) if isinstance(spec, str) else spec
    if spec.provider == "ollama":
        return _ollama_running()
    return bool(OPENAI_API_KEY if spec.provider == "openai" else ANTHROPIC_API_KEY)


def _openai(spec: ModelSpec, instructions: str, prompt: str) -> Completion:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    # 추론 모델은 추론 토큰이 출력 한도에 포함되므로 한도를 넉넉히 준다
    extra = (
        {"reasoning": {"effort": spec.effort}, "max_output_tokens": ANTHROPIC_MAX_TOKENS}
        if spec.effort
        else {"max_output_tokens": MAX_OUTPUT_TOKENS}
    )
    started = time.perf_counter()
    response = client.responses.create(
        model=spec.model,
        instructions=instructions,
        input=prompt,
        **extra,
    )
    latency = time.perf_counter() - started
    usage = response.usage
    return Completion(
        text=(response.output_text or "").strip(),
        model=getattr(response, "model", spec.model),
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        latency_s=latency,
        stop_reason=getattr(response, "status", "") or "",
    )


def _anthropic(spec: ModelSpec, instructions: str, prompt: str) -> Completion:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    extra = {"output_config": {"effort": spec.effort}} if spec.effort else {}
    started = time.perf_counter()
    # 안전 분류기가 요청을 거절하면 서버가 권장 모델로 다시 돌린다.
    # 비교에서는 어느 모델이 실제로 답했는지(response.model)를 함께 기록한다.
    response = client.beta.messages.create(
        model=spec.model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=instructions,
        messages=[{"role": "user", "content": prompt}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        **extra,
    )
    latency = time.perf_counter() - started
    fallback_used = any(
        getattr(entry, "type", "") == "fallback_message"
        for entry in (getattr(response.usage, "iterations", None) or [])
    )
    if response.stop_reason == "refusal":
        text = ""
    else:
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
    return Completion(
        text=text,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_s=latency,
        stop_reason=response.stop_reason or "",
        fallback_used=fallback_used,
    )


def _ollama(spec: ModelSpec, instructions: str, prompt: str) -> Completion:
    """로컬 모델. Ollama 기본 API 를 쓴다 (사고 모드를 켜고 끌 수 있다)."""
    import httpx

    body = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        # 같은 질문에 같은 답이 나오게 한다 (평가 재현)
        "options": {"temperature": 0, "seed": 7},
    }
    if spec.effort in {"think", "nothink"}:
        body["think"] = spec.effort == "think"
    started = time.perf_counter()
    response = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=600.0)
    response.raise_for_status()
    data = response.json()
    latency = time.perf_counter() - started
    return Completion(
        text=(data.get("message", {}).get("content") or "").strip(),
        model=data.get("model", spec.model),
        input_tokens=data.get("prompt_eval_count", 0) or 0,
        output_tokens=data.get("eval_count", 0) or 0,
        latency_s=latency,
        stop_reason=data.get("done_reason", "") or "",
    )


Generator = Callable[[str, str, str], Completion]


def complete(spec: str, instructions: str, prompt: str) -> Completion:
    parsed = parse_spec(spec)
    if not has_key(parsed):
        if parsed.provider == "ollama":
            raise RuntimeError(f"Ollama 서버({OLLAMA_URL})에 연결할 수 없습니다.")
        name = "OPENAI_API_KEY" if parsed.provider == "openai" else "ANTHROPIC_API_KEY"
        raise RuntimeError(f"{name} 가 .env 에 없습니다.")
    if parsed.provider == "openai":
        return _openai(parsed, instructions, prompt)
    if parsed.provider == "ollama":
        return _ollama(parsed, instructions, prompt)
    return _anthropic(parsed, instructions, prompt)
