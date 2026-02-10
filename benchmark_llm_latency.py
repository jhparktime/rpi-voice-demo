#!/usr/bin/env python3
"""
LLM Latency Benchmark: Compare Ollama vs Gemini API response times.

Usage:
    python benchmark_llm_latency.py
    python benchmark_llm_latency.py --iterations 10
    python benchmark_llm_latency.py --ollama-only
    python benchmark_llm_latency.py --gemini-only
"""

import os
import sys
import time
import argparse
from statistics import mean, median, stdev
from typing import List, Dict, Optional, Tuple

# Import existing modules
from Demo import llm_ollama
from Demo import cloud_llm


# Test queries with different characteristics
TEST_QUERIES = [
    {
        "name": "Simple fact",
        "prompt": "What is the capital of France?",
        "type": "simple"
    },
    {
        "name": "Conversational",
        "prompt": "How are you today?",
        "type": "conversation"
    },
    {
        "name": "Complex reasoning",
        "prompt": "Explain why the sky appears blue during the day in simple terms.",
        "type": "complex"
    },
    {
        "name": "Math",
        "prompt": "What is 127 multiplied by 43?",
        "type": "simple"
    },
    {
        "name": "Emotional",
        "prompt": "I'm feeling stressed about my exams.",
        "type": "conversation"
    },
]


def benchmark_ollama(
    prompt: str,
    model: str = "smollm2:360m",
    url: str = "http://localhost:11434/api/generate",
) -> Tuple[float, str]:
    """Benchmark Ollama latency for a single query.
    
    Returns (latency_seconds, response_text).
    """
    system = "You are a helpful assistant. Reply in 1-2 short sentences."
    
    t_start = time.perf_counter()
    try:
        response = llm_ollama.generate_ollama(
            prompt=prompt,
            model=model,
            system=system,
            url=url,
            num_predict=50,
            temperature=0.7,
            timeout=30,
            max_sentences=2,
            max_words=50,
        )
    except Exception as exc:
        return -1.0, f"ERROR: {exc}"
    t_end = time.perf_counter()
    
    latency = t_end - t_start
    return latency, response


def benchmark_gemini(prompt: str) -> Tuple[float, str]:
    """Benchmark Gemini API latency for a single query.
    
    Returns (latency_seconds, response_text).
    """
    system = "You are a helpful assistant. Reply in 1-2 short sentences."
    
    t_start = time.perf_counter()
    try:
        response = cloud_llm.call_cloud_llm(
            prompt=prompt,
            system=system,
            timeout=30,
        )
    except Exception as exc:
        return -1.0, f"ERROR: {exc}"
    t_end = time.perf_counter()
    
    latency = t_end - t_start
    return latency, response


def print_statistics(name: str, latencies: List[float], responses: List[str] = None) -> None:
    """Print statistical summary of latencies."""
    if not latencies or all(l < 0 for l in latencies):
        print(f"  {name}: ALL FAILED")
        return
    
    valid = [l for l in latencies if l >= 0]
    if not valid:
        print(f"  {name}: NO VALID DATA")
        return
    
    # Check for rate limit errors
    rate_limited = 0
    if responses:
        rate_limited = sum(1 for r in responses if "429" in r or "rate limit" in r.lower())
    
    print(f"  {name}:")
    print(f"    Min:    {min(valid):.3f}s")
    print(f"    Max:    {max(valid):.3f}s")
    print(f"    Mean:   {mean(valid):.3f}s")
    print(f"    Median: {median(valid):.3f}s")
    if len(valid) > 1:
        print(f"    StdDev: {stdev(valid):.3f}s")
    print(f"    Successes: {len(valid)}/{len(latencies)}")
    if rate_limited > 0:
        print(f"    ⚠️ Rate limited: {rate_limited}/{len(latencies)} requests")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama vs Gemini latency")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per query")
    parser.add_argument("--ollama-only", action="store_true", help="Test Ollama only")
    parser.add_argument("--gemini-only", action="store_true", help="Test Gemini only")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--ollama-model", default="smollm2:360m")
    args = parser.parse_args()
    
    # Check environment
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_cloud = bool(os.environ.get("CLOUD_LLM_URL"))
    
    print("="*70)
    print("LLM LATENCY BENCHMARK")
    print("="*70)
    print(f"Iterations per query: {args.iterations}")
    print(f"Test queries: {len(TEST_QUERIES)}")
    print()
    
    if not args.gemini_only:
        print(f"[Ollama] Model: {args.ollama_model}")
        print(f"[Ollama] URL: {args.ollama_url}")
    
    if not args.ollama_only:
        if has_gemini:
            print("[Gemini] API key found")
            print("[Gemini] Free tier limits: 15 RPM, 1M TPM, 1500 RPD")
            print("[Gemini] Strategy: Interleave with Ollama + enforce 4s minimum interval")
        elif has_openai:
            print("[OpenAI] API key found")
        elif has_cloud:
            print(f"[Cloud] Custom URL: {os.environ.get('CLOUD_LLM_URL')}")
        else:
            print("[WARNING] No Cloud LLM configured (set GEMINI_API_KEY or OPENAI_API_KEY)")
    
    print("="*70)
    
    # Ollama warm-up (if testing Ollama)
    if not args.gemini_only:
        print("\n[Ollama] Warming up model (first call loads model into memory)...")
        warmup_start = time.perf_counter()
        _, warmup_response = benchmark_ollama(
            "Hello",
            model=args.ollama_model,
            url=args.ollama_url,
        )
        warmup_end = time.perf_counter()
        print(f"[Ollama] Warm-up completed in {warmup_end - warmup_start:.2f}s")
        print(f"[Ollama] Response: {warmup_response[:50]}...")
    
    print()
    
    # Results storage
    results: Dict[str, Dict[str, List[float]]] = {
        "ollama": {},
        "gemini": {},
    }
    results_responses: Dict[str, Dict[str, List[str]]] = {
        "ollama": {},
        "gemini": {},
    }
    
    # Rate limit management for Gemini (15 RPM = 4 seconds per request)
    GEMINI_MIN_INTERVAL = 4.0  # seconds
    last_gemini_call_time: Optional[float] = None
    
    # Run benchmarks
    for query in TEST_QUERIES:
        print(f"\n{'─'*70}")
        print(f"Query: {query['name']} ({query['type']})")
        print(f"Prompt: \"{query['prompt']}\"")
        print(f"{'─'*70}")
        
        ollama_latencies: List[float] = []
        gemini_latencies: List[float] = []
        ollama_responses: List[str] = []
        gemini_responses: List[str] = []
        
        # Interleaved benchmark: Ollama, Gemini, Ollama, Gemini, ...
        # This naturally spaces out Gemini calls to avoid rate limits
        print(f"\n[Interleaved] Running {args.iterations} iteration(s) (Ollama → Gemini → Ollama → ...)...")
        
        for i in range(args.iterations):
            # Ollama
            if not args.gemini_only:
                latency, response = benchmark_ollama(
                    query["prompt"],
                    model=args.ollama_model,
                    url=args.ollama_url,
                )
                ollama_latencies.append(latency)
                ollama_responses.append(response)
                
                status = f"{latency:.3f}s" if latency >= 0 else "FAILED"
                response_preview = response[:60] + "..." if len(response) > 60 else response
                print(f"  [Ollama {i+1}/{args.iterations}] {status} | {response_preview}")
            
            # Gemini (interleaved after Ollama to avoid rate limits)
            if not args.ollama_only and (has_gemini or has_openai or has_cloud):
                # Rate limit management: ensure at least 4 seconds between Gemini calls
                if last_gemini_call_time is not None:
                    time_since_last = time.perf_counter() - last_gemini_call_time
                    if time_since_last < GEMINI_MIN_INTERVAL:
                        wait_time = GEMINI_MIN_INTERVAL - time_since_last
                        print(f"  [Rate limit] Waiting {wait_time:.1f}s before next Gemini call...")
                        time.sleep(wait_time)
                
                call_start = time.perf_counter()
                latency, response = benchmark_gemini(query["prompt"])
                last_gemini_call_time = time.perf_counter()
                
                gemini_latencies.append(latency)
                gemini_responses.append(response)
                
                status = f"{latency:.3f}s" if latency >= 0 else "FAILED"
                response_preview = response[:60] + "..." if len(response) > 60 else response
                
                # Check for rate limit error
                if "429" in response or "rate limit" in response.lower():
                    print(f"  [Gemini {i+1}/{args.iterations}] {status} | {response_preview} ⚠️ RATE LIMITED")
                else:
                    print(f"  [Gemini {i+1}/{args.iterations}] {status} | {response_preview}")
        
        # Store results
        results["ollama"][query["name"]] = ollama_latencies
        results["gemini"][query["name"]] = gemini_latencies
        results_responses["ollama"][query["name"]] = ollama_responses
        results_responses["gemini"][query["name"]] = gemini_responses
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    for query in TEST_QUERIES:
        print(f"{query['name']} ({query['type']}):")
        
        if not args.gemini_only and results["ollama"][query["name"]]:
            print_statistics("Ollama", results["ollama"][query["name"]], results_responses["ollama"][query["name"]])
        
        if not args.ollama_only and results["gemini"][query["name"]]:
            print_statistics("Gemini/Cloud", results["gemini"][query["name"]], results_responses["gemini"][query["name"]])
        
        # Comparison
        ollama_valid = [l for l in results["ollama"].get(query["name"], []) if l >= 0]
        gemini_valid = [l for l in results["gemini"].get(query["name"], []) if l >= 0]
        
        if ollama_valid and gemini_valid:
            ollama_avg = mean(ollama_valid)
            gemini_avg = mean(gemini_valid)
            speedup = ollama_avg / gemini_avg if gemini_avg > 0 else 0
            faster = "Gemini" if gemini_avg < ollama_avg else "Ollama"
            print(f"  Winner: {faster} ({speedup:.2f}x faster)")
        
        print()
    
    # Overall winner
    print(f"{'='*70}")
    print("OVERALL COMPARISON")
    print(f"{'='*70}\n")
    
    all_ollama = [l for latencies in results["ollama"].values() for l in latencies if l >= 0]
    all_gemini = [l for latencies in results["gemini"].values() for l in latencies if l >= 0]
    
    if all_ollama:
        print(f"Ollama Overall: {mean(all_ollama):.3f}s average ({len(all_ollama)} samples)")
    if all_gemini:
        print(f"Gemini Overall: {mean(all_gemini):.3f}s average ({len(all_gemini)} samples)")
    
    if all_ollama and all_gemini:
        speedup = mean(all_ollama) / mean(all_gemini)
        faster = "Gemini" if mean(all_gemini) < mean(all_ollama) else "Ollama"
        print(f"\nOverall Winner: {faster} ({speedup:.2f}x)")
    
    print()


if __name__ == "__main__":
    main()
