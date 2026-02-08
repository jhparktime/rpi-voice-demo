"""ONNX CausalLM (e.g. SmolLM2-135M-Instruct) load and generate."""
from __future__ import annotations

import sys
from typing import Any

from . import text_utils


def _load_onnx_llm(model_id: str):
    """Load ONNX CausalLM and tokenizer (optimum-onnxruntime). Returns (model, tokenizer) or (None, None) on failure."""
    try:
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError as e:
        print(
            f"[onnx-llm] Optional dependency missing: {e}. Install with: pip install optimum[onnxruntime]",
            file=sys.stderr,
        )
        return None, None
    try:
        model = ORTModelForCausalLM.from_pretrained(model_id, subfolder="onnx", provider="CPUExecutionProvider")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        print(f"[onnx-llm] Load failed: {e}", file=sys.stderr)
        return None, None


def generate_onnx_llm(
    prompt: str,
    system: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 24,
    temperature: float = 0.3,
    max_sentences: int = 2,
    max_words: int = 36,
) -> str:
    """Generate reply using ONNX CausalLM (e.g. SmolLM2-135M-Instruct). Returns postprocessed text."""
    if not (prompt or "").strip() or model is None or tokenizer is None:
        return ""
    system = (system or text_utils.ONNX_DEFAULT_SYSTEM).strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt.strip()},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(text, list):
            text = tokenizer.decode(text, skip_special_tokens=False)
        inputs = tokenizer(text, return_tensors="pt")
        if hasattr(model, "generate"):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
        else:
            return "(ONNX LLM: no generate method)"
        out = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return text_utils.postprocess_output(out.strip(), max_sentences=max_sentences, max_words=max_words)
    except Exception as e:
        return f"(ONNX LLM error: {e})"
