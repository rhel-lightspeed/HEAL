#!/usr/bin/env python3
"""Analyze correlation between fast metrics (F1, MRR) and RAGAS metrics.

This helps answer: "Can we trust F1/MRR improvements as proxies for answer quality?"

Usage:
    # Analyze single pattern
    python scripts/analyze_metric_correlations.py BOOTLOADER_GRUB_ISSUES

    # Analyze all patterns
    python scripts/analyze_metric_correlations.py --all

    # Generate report
    python scripts/analyze_metric_correlations.py --all --report correlations.md

    # Use existing evaluation results (not just iteration files)
    python scripts/analyze_metric_correlations.py --all --use-evals
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_iterations_from_jsonl(pattern_id: str) -> Optional[pd.DataFrame]:
    """Load iteration data from pattern database JSONL file."""
    # Try .claude/fix_patterns first (current location)
    iterations_file = Path(f".claude/fix_patterns/{pattern_id}_iterations.jsonl")

    # Fallback to old location for backward compatibility
    if not iterations_file.exists():
        iterations_file = Path(f".diagnostics/{pattern_id}/{pattern_id}_iterations.jsonl")

    if not iterations_file.exists():
        return None

    # Load JSONL
    iterations = []
    with open(iterations_file) as f:
        for line in f:
            try:
                iterations.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not iterations:
        return None

    return pd.DataFrame(iterations)


def load_validation_checkpoints(pattern_id: str) -> Optional[pd.DataFrame]:
    """Load per-ticket validation checkpoint data with F1 and answer_correctness."""
    # Check for validation checkpoints file
    validation_file = Path(f".claude/fix_patterns/{pattern_id}_validation_checkpoints.jsonl")

    if not validation_file.exists():
        return None

    # Load JSONL
    checkpoints = []
    with open(validation_file) as f:
        for line in f:
            try:
                checkpoints.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not checkpoints:
        return None

    return pd.DataFrame(checkpoints)


def load_eval_results(pattern_id: str) -> Optional[pd.DataFrame]:
    """Load evaluation results from .diagnostics directory.

    This finds baseline and validation results from existing runs.
    """
    diagnostics_dir = Path(f".diagnostics/{pattern_id}")

    if not diagnostics_dir.exists():
        return None

    # Look for evaluation result JSON files
    eval_files = list(diagnostics_dir.glob("*_result*.json")) + \
                 list(diagnostics_dir.glob("baseline*.json")) + \
                 list(diagnostics_dir.glob("validation*.json"))

    if not eval_files:
        return None

    records = []

    for eval_file in eval_files:
        try:
            with open(eval_file) as f:
                data = json.load(f)

            # Extract metrics from various result formats
            record = {
                'source': eval_file.name,
                'timestamp': eval_file.stat().st_mtime,
            }

            # Handle PatternEvaluationResult format
            if 'pattern_url_f1' in data:
                record.update({
                    'url_f1': data.get('pattern_url_f1'),
                    'mrr': data.get('pattern_mrr'),
                    'context_relevance': data.get('pattern_context_relevance'),
                    'context_precision': data.get('pattern_context_precision'),
                    'answer_correctness': data.get('pattern_answer_correctness'),
                    'faithfulness': data.get('pattern_faithfulness'),
                    'response_relevancy': data.get('pattern_response_relevancy'),
                    'docs_retrieved': data.get('pattern_docs_retrieved'),
                })
            # Handle EvaluationResult format
            elif 'url_f1' in data:
                record.update({
                    'url_f1': data.get('url_f1'),
                    'mrr': data.get('mrr'),
                    'context_relevance': data.get('context_relevance'),
                    'context_precision': data.get('context_precision'),
                    'answer_correctness': data.get('answer_correctness'),
                    'faithfulness': data.get('faithfulness'),
                    'response_relevancy': data.get('response_relevancy'),
                    'docs_retrieved': data.get('docs_retrieved'),
                })
            # Handle nested final_metrics format
            elif 'final_metrics' in data:
                metrics = data['final_metrics']
                record.update({
                    'url_f1': metrics.get('url_f1'),
                    'mrr': metrics.get('mrr'),
                    'context_relevance': metrics.get('context_relevance'),
                    'context_precision': metrics.get('context_precision'),
                    'answer_correctness': metrics.get('answer_correctness'),
                    'faithfulness': metrics.get('faithfulness'),
                    'response_relevancy': metrics.get('response_relevancy'),
                })

            # Only add if we have at least some metrics
            if any(v is not None for k, v in record.items() if k not in ['source', 'timestamp']):
                records.append(record)

        except Exception as e:
            print(f"⚠️  Skipping {eval_file.name}: {e}")
            continue

    if not records:
        return None

    df = pd.DataFrame(records)

    # Sort by timestamp
    df = df.sort_values('timestamp')

    return df


def load_pattern_data(pattern_id: str, use_evals: bool = False) -> Optional[pd.DataFrame]:
    """Load data for a pattern from all available sources.

    Priority:
    1. Validation checkpoints (per-ticket, has F1 + answer_correctness)
    2. Iteration JSONL (per-ticket, has F1 only)
    3. Eval results (if use_evals=True)
    """

    dfs = []

    # PRIORITY 1: Try validation checkpoints (has both F1 and answer data per ticket)
    df_validation = load_validation_checkpoints(pattern_id)
    if df_validation is not None:
        df_validation['source_type'] = 'validation_checkpoint'
        dfs.append(df_validation)
        print(f"   Loaded {len(df_validation)} per-ticket validation checkpoints")
        print(f"   Tickets tracked: {df_validation['ticket_id'].nunique()}")

    # PRIORITY 2: Try to load iteration JSONL (fast metrics only)
    if not dfs:  # Only load if no validation data
        df_iterations = load_iterations_from_jsonl(pattern_id)
        if df_iterations is not None:
            df_iterations['source_type'] = 'iteration'
            dfs.append(df_iterations)
            print(f"   Loaded {len(df_iterations)} iterations from JSONL")

    # PRIORITY 3: Optionally load eval results
    if use_evals and not dfs:  # Only if nothing else found
        df_evals = load_eval_results(pattern_id)
        if df_evals is not None:
            df_evals['source_type'] = 'eval'
            dfs.append(df_evals)
            print(f"   Loaded {len(df_evals)} evaluation results")

    if not dfs:
        return None

    # Combine all sources
    df = pd.concat(dfs, ignore_index=True)

    # Normalize column names (validation checkpoints use "current_*", others use direct names)
    if 'current_url_f1' in df.columns and 'url_f1' not in df.columns:
        df['url_f1'] = df['current_url_f1']
    if 'current_answer_correctness' in df.columns and 'answer_correctness' not in df.columns:
        df['answer_correctness'] = df['current_answer_correctness']
    if 'current_mrr' in df.columns and 'mrr' not in df.columns:
        df['mrr'] = df['current_mrr']

    # For per-ticket data, calculate deltas within each ticket
    if 'ticket_id' in df.columns:
        # Group by ticket and calculate deltas
        for metric in ['url_f1', 'mrr', 'answer_correctness', 'context_relevance', 'faithfulness']:
            if metric in df.columns:
                df[f'{metric}_delta'] = df.groupby('ticket_id')[metric].diff()
    else:
        # Pattern-averaged data: calculate global deltas
        for metric in ['url_f1', 'mrr', 'answer_correctness', 'context_relevance', 'faithfulness']:
            if metric in df.columns:
                df[f'{metric}_delta'] = df[metric].diff()

    return df


def analyze_per_ticket_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze correlations for each ticket individually.

    Returns:
        Dict with:
        - per_ticket_correlations: {ticket_id: {metric_pair: (r, p)}}
        - aggregate_stats: Summary statistics across tickets
    """
    if 'ticket_id' not in df.columns:
        return None

    per_ticket_corrs = {}

    for ticket_id in df['ticket_id'].unique():
        ticket_data = df[df['ticket_id'] == ticket_id]

        # Need at least 3 data points to calculate correlation
        complete = ticket_data.dropna(subset=['current_url_f1', 'current_answer_correctness'])

        if len(complete) < 2:
            continue

        try:
            r, p = stats.pearsonr(complete['current_url_f1'], complete['current_answer_correctness'])
            per_ticket_corrs[ticket_id] = {
                'f1_vs_answer': (r, p),
                'n_samples': len(complete),
            }

            # Delta correlation if available
            deltas = ticket_data.dropna(subset=['url_f1_delta', 'answer_correctness_delta'])
            if len(deltas) >= 2:
                r_delta, p_delta = stats.pearsonr(deltas['url_f1_delta'], deltas['answer_correctness_delta'])
                per_ticket_corrs[ticket_id]['f1_delta_vs_answer_delta'] = (r_delta, p_delta)

        except Exception as e:
            continue

    # Aggregate statistics
    if per_ticket_corrs:
        all_r_values = [v['f1_vs_answer'][0] for v in per_ticket_corrs.values()]
        aggregate_stats = {
            'mean_r': np.mean(all_r_values),
            'median_r': np.median(all_r_values),
            'std_r': np.std(all_r_values),
            'min_r': np.min(all_r_values),
            'max_r': np.max(all_r_values),
            'n_tickets': len(per_ticket_corrs),
            'strong_correlation_tickets': sum(1 for r in all_r_values if r > 0.7),
            'moderate_correlation_tickets': sum(1 for r in all_r_values if 0.4 < r <= 0.7),
            'weak_correlation_tickets': sum(1 for r in all_r_values if r <= 0.4),
        }

        return {
            'per_ticket_correlations': per_ticket_corrs,
            'aggregate_stats': aggregate_stats,
        }

    return None


def calculate_correlations(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """Calculate correlation coefficients between metrics.

    Returns:
        Dict mapping correlation name to (r, p_value) tuple
    """

    # Filter to rows with BOTH fast and judge metrics
    complete = df.dropna(subset=['url_f1', 'answer_correctness'])

    if len(complete) < 3:
        print("⚠️  Insufficient data for correlation (need ≥3 complete samples)")
        return {}

    correlations = {}

    # Absolute values
    try:
        correlations['f1_vs_answer'] = stats.pearsonr(
            complete['url_f1'], complete['answer_correctness']
        )
    except Exception:
        pass

    try:
        correlations['mrr_vs_answer'] = stats.pearsonr(
            complete['mrr'], complete['answer_correctness']
        )
    except Exception:
        pass

    # F1 vs context metrics
    if 'context_relevance' in complete.columns:
        try:
            complete_ctx = complete.dropna(subset=['context_relevance'])
            if len(complete_ctx) >= 3:
                correlations['f1_vs_context_rel'] = stats.pearsonr(
                    complete_ctx['url_f1'], complete_ctx['context_relevance']
                )
        except Exception:
            pass

    if 'context_precision' in complete.columns:
        try:
            complete_prec = complete.dropna(subset=['context_precision'])
            if len(complete_prec) >= 3:
                correlations['f1_vs_context_prec'] = stats.pearsonr(
                    complete_prec['url_f1'], complete_prec['context_precision']
                )
        except Exception:
            pass

    # Deltas (more important: do CHANGES correlate?)
    deltas = complete.dropna(subset=['url_f1_delta', 'answer_correctness_delta'])
    if len(deltas) >= 3:
        try:
            correlations['f1_delta_vs_answer_delta'] = stats.pearsonr(
                deltas['url_f1_delta'], deltas['answer_correctness_delta']
            )
        except Exception:
            pass

    # Only calculate MRR delta correlation if MRR data exists
    if 'mrr_delta' in complete.columns:
        deltas_mrr = complete.dropna(subset=['mrr_delta', 'answer_correctness_delta'])
        if len(deltas_mrr) >= 3:
            try:
                correlations['mrr_delta_vs_answer_delta'] = stats.pearsonr(
                    deltas_mrr['mrr_delta'], deltas_mrr['answer_correctness_delta']
                )
            except Exception:
                pass

    return correlations


def plot_correlations(df: pd.DataFrame, pattern_id: str, output_dir: Path):
    """Generate correlation visualizations."""

    complete = df.dropna(subset=['url_f1', 'answer_correctness'])

    if len(complete) < 3:
        print("⚠️  Skipping plots (insufficient data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Metric Correlations: {pattern_id}', fontsize=16, fontweight='bold')

    # 1. F1 vs Answer Correctness
    ax = axes[0, 0]
    ax.scatter(complete['url_f1'], complete['answer_correctness'], alpha=0.6, s=60)
    ax.set_xlabel('URL F1 (deterministic)', fontsize=11)
    ax.set_ylabel('Answer Correctness (LLM judge)', fontsize=11)
    ax.set_title('F1 vs Answer Correctness', fontweight='bold')
    ax.grid(alpha=0.3)

    # Add regression line
    if len(complete) >= 2:
        z = np.polyfit(complete['url_f1'], complete['answer_correctness'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(complete['url_f1'].min(), complete['url_f1'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Add correlation coefficient
    try:
        r, p_val = stats.pearsonr(complete['url_f1'], complete['answer_correctness'])

        # Color code by strength
        if r > 0.7:
            color = 'green'
            interpretation = 'STRONG'
        elif r > 0.4:
            color = 'orange'
            interpretation = 'MODERATE'
        else:
            color = 'red'
            interpretation = 'WEAK'

        ax.text(0.05, 0.95, f'r = {r:.3f} ({interpretation})\np = {p_val:.3f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor=color))
    except Exception:
        pass

    # 2. MRR vs Answer Correctness
    ax = axes[0, 1]
    complete_mrr = complete.dropna(subset=['mrr'])
    if len(complete_mrr) >= 3:
        ax.scatter(complete_mrr['mrr'], complete_mrr['answer_correctness'], alpha=0.6, s=60)
        ax.set_xlabel('MRR (deterministic)', fontsize=11)
        ax.set_ylabel('Answer Correctness (LLM judge)', fontsize=11)
        ax.set_title('MRR vs Answer Correctness', fontweight='bold')
        ax.grid(alpha=0.3)

        if len(complete_mrr) >= 2:
            z = np.polyfit(complete_mrr['mrr'], complete_mrr['answer_correctness'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(complete_mrr['mrr'].min(), complete_mrr['mrr'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        try:
            r, p_val = stats.pearsonr(complete_mrr['mrr'], complete_mrr['answer_correctness'])

            if r > 0.7:
                color = 'green'
                interpretation = 'STRONG'
            elif r > 0.4:
                color = 'orange'
                interpretation = 'MODERATE'
            else:
                color = 'red'
                interpretation = 'WEAK'

            ax.text(0.05, 0.95, f'r = {r:.3f} ({interpretation})\np = {p_val:.3f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor=color))
        except Exception:
            pass
    else:
        ax.text(0.5, 0.5, 'Insufficient MRR data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_xlabel('MRR')
        ax.set_ylabel('Answer Correctness')

    # 3. F1 Delta vs Answer Delta
    ax = axes[1, 0]
    deltas = complete.dropna(subset=['url_f1_delta', 'answer_correctness_delta'])
    if len(deltas) >= 3:
        ax.scatter(deltas['url_f1_delta'], deltas['answer_correctness_delta'], alpha=0.6, s=60)
        ax.set_xlabel('Δ URL F1', fontsize=11)
        ax.set_ylabel('Δ Answer Correctness', fontsize=11)
        ax.set_title('Change in F1 vs Change in Answer', fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(alpha=0.3)

        try:
            r, p_val = stats.pearsonr(deltas['url_f1_delta'], deltas['answer_correctness_delta'])

            if r > 0.7:
                color = 'green'
                interpretation = 'STRONG'
            elif r > 0.4:
                color = 'orange'
                interpretation = 'MODERATE'
            else:
                color = 'red'
                interpretation = 'WEAK'

            ax.text(0.05, 0.95, f'r = {r:.3f} ({interpretation})\np = {p_val:.3f}',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor=color))
        except Exception:
            pass
    else:
        ax.text(0.5, 0.5, 'Insufficient delta data\n(need multiple iterations)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlabel('Δ URL F1')
        ax.set_ylabel('Δ Answer Correctness')

    # 4. Correlation Heatmap
    ax = axes[1, 1]

    # Build correlation matrix from available metrics
    available_metrics = []
    for metric in ['url_f1', 'mrr', 'context_relevance', 'context_precision',
                   'answer_correctness', 'faithfulness', 'response_relevancy']:
        if metric in complete.columns and complete[metric].notna().sum() >= 3:
            available_metrics.append(metric)

    if len(available_metrics) >= 2:
        corr_data = complete[available_metrics].dropna()
        if len(corr_data) >= 3:
            corr_matrix = corr_data.corr()

            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, vmin=-1, vmax=1, ax=ax,
                        cbar_kws={'label': 'Correlation'}, square=True)
            ax.set_title('Metric Correlation Matrix', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient complete data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, f'Insufficient metrics\n({len(available_metrics)} available)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    plt.tight_layout()

    # Save
    output_file = output_dir / f'{pattern_id}_correlations.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved correlation plots: {output_file}")
    plt.close()


def generate_report(all_patterns_data: Dict[str, pd.DataFrame], output_file: Path):
    """Generate markdown report summarizing correlations across all patterns."""

    report = """# Metric Correlation Analysis Report

## Executive Summary

This report analyzes the correlation between:
- **Fast, deterministic metrics**: URL F1, MRR (no LLM judge, ~30 sec, $0 tokens)
- **LLM-judged RAGAS metrics**: answer_correctness, context_relevance, faithfulness (~20 min, thousands of tokens)

**Key Question**: Can we trust F1/MRR improvements as proxies for answer quality improvements?

**Why this matters**:
- If correlation is STRONG → Optimize for F1 (cheap), validate RAGAS rarely
- If correlation is WEAK → Must use RAGAS metrics directly (expensive)

---

## Findings

"""

    # Aggregate correlations across all patterns
    all_f1_answer_corrs = []
    all_mrr_answer_corrs = []
    all_f1_delta_corrs = []
    pattern_details = []

    for pattern_id, df in all_patterns_data.items():
        complete = df.dropna(subset=['url_f1', 'answer_correctness'])

        if len(complete) < 3:
            continue

        try:
            r_f1, p_f1 = stats.pearsonr(complete['url_f1'], complete['answer_correctness'])
            all_f1_answer_corrs.append(r_f1)

            pattern_details.append({
                'pattern_id': pattern_id,
                'r_f1': r_f1,
                'p_f1': p_f1,
                'n_samples': len(complete),
            })
        except Exception:
            pass

        try:
            complete_mrr = complete.dropna(subset=['mrr'])
            if len(complete_mrr) >= 3:
                r_mrr, _ = stats.pearsonr(complete_mrr['mrr'], complete_mrr['answer_correctness'])
                all_mrr_answer_corrs.append(r_mrr)
        except Exception:
            pass

        try:
            deltas = complete.dropna(subset=['url_f1_delta', 'answer_correctness_delta'])
            if len(deltas) >= 3:
                r_delta, _ = stats.pearsonr(deltas['url_f1_delta'], deltas['answer_correctness_delta'])
                all_f1_delta_corrs.append(r_delta)
        except Exception:
            pass

    # Calculate aggregate statistics
    if all_f1_answer_corrs:
        report += f"""### Aggregate Correlation Strength

| Metric Pair | Mean r | Median r | Min r | Max r | N Patterns |
|-------------|--------|----------|-------|-------|------------|
| F1 vs Answer Correctness | {np.mean(all_f1_answer_corrs):.3f} | {np.median(all_f1_answer_corrs):.3f} | {np.min(all_f1_answer_corrs):.3f} | {np.max(all_f1_answer_corrs):.3f} | {len(all_f1_answer_corrs)} |
"""

        if all_mrr_answer_corrs:
            report += f"| MRR vs Answer Correctness | {np.mean(all_mrr_answer_corrs):.3f} | {np.median(all_mrr_answer_corrs):.3f} | {np.min(all_mrr_answer_corrs):.3f} | {np.max(all_mrr_answer_corrs):.3f} | {len(all_mrr_answer_corrs)} |\n"

        if all_f1_delta_corrs:
            report += f"| ΔF1 vs ΔAnswer | {np.mean(all_f1_delta_corrs):.3f} | {np.median(all_f1_delta_corrs):.3f} | {np.min(all_f1_delta_corrs):.3f} | {np.max(all_f1_delta_corrs):.3f} | {len(all_f1_delta_corrs)} |\n"

        report += """
**Interpretation**:
- r > 0.7: **STRONG** correlation → F1 is a good proxy for answer quality
- 0.4 < r < 0.7: **MODERATE** correlation → F1 partially predicts answer quality
- r < 0.4: **WEAK** correlation → Cannot trust F1 alone, must use RAGAS

"""

        # Overall recommendation
        mean_r = np.mean(all_f1_answer_corrs)
        if mean_r > 0.7:
            report += f"""### 🎉 Overall Recommendation: **FAST OPTIMIZATION**

With mean correlation r = {mean_r:.3f}, F1 improvements strongly predict answer quality improvements.

**Suggested strategy**:
- Inner loop: Optimize for F1/MRR only (fast, deterministic)
- Validation: Run RAGAS every 3-5 cycles or only at the end
- **Cost savings**: ~75% reduction in RAGAS validation time/tokens

"""
        elif mean_r > 0.4:
            report += f"""### ⚠️  Overall Recommendation: **HYBRID OPTIMIZATION**

With mean correlation r = {mean_r:.3f}, F1 improvements moderately predict answer quality.

**Suggested strategy**:
- Inner loop: Optimize for F1/MRR
- Validation: Run RAGAS every 2-3 cycles
- **Cost savings**: ~50% reduction in RAGAS validation time/tokens

"""
        else:
            report += f"""### ❌ Overall Recommendation: **DIRECT RAGAS OPTIMIZATION**

With mean correlation r = {mean_r:.3f}, F1 improvements do NOT reliably predict answer quality.

**Suggested strategy**:
- Inner loop: Use quick RAGAS sampling (1 run instead of 3-6)
- Validation: Run full RAGAS every cycle
- **Cost savings**: Minimal, but necessary to get meaningful feedback

"""

    else:
        report += "❌ Insufficient data across all patterns to calculate aggregate statistics.\n\n"

    # Per-pattern breakdown
    report += "\n---\n\n## Per-Pattern Breakdown\n\n"

    # Sort patterns by correlation strength (strongest first)
    pattern_details.sort(key=lambda x: x['r_f1'], reverse=True)

    for detail in pattern_details:
        pattern_id = detail['pattern_id']
        r_f1 = detail['r_f1']
        p_f1 = detail['p_f1']
        n_samples = detail['n_samples']

        # Determine strength
        if r_f1 > 0.7:
            strength = "✅ STRONG"
            color_emoji = "🟢"
            recommendation = "Trust F1 improvements → validate RAGAS rarely"
        elif r_f1 > 0.4:
            strength = "⚠️  MODERATE"
            color_emoji = "🟡"
            recommendation = "F1 partially predictive → validate RAGAS every 2-3 cycles"
        else:
            strength = "❌ WEAK"
            color_emoji = "🔴"
            recommendation = "F1 does NOT predict answer quality → validate RAGAS every cycle"

        report += f"""### {color_emoji} {pattern_id}

- **Correlation**: r = {r_f1:.3f} (p = {p_f1:.3f}) - {strength}
- **Samples**: {n_samples}
- **Recommendation**: {recommendation}

"""

    # Patterns with insufficient data
    insufficient = []
    for pattern_id, df in sorted(all_patterns_data.items()):
        complete = df.dropna(subset=['url_f1', 'answer_correctness'])
        if len(complete) < 3:
            insufficient.append(f"  - {pattern_id}: {len(complete)} samples (need ≥3)")

    if insufficient:
        report += "### Patterns with Insufficient Data\n\n"
        report += "\n".join(insufficient)
        report += "\n\n"

    # Recommendations section
    report += """---

## Implementation Recommendations

Based on these findings, implement the following optimization strategies:

### 1. Add `--validation-strategy` flag to fix.sh

```bash
# Strong correlation patterns
./runners/fix.sh PATTERN_ID --validation-strategy fast --validation-cycles 5

# Moderate correlation patterns
./runners/fix.sh PATTERN_ID --validation-strategy adaptive --validation-cycles 3

# Weak correlation patterns
./runners/fix.sh PATTERN_ID --validation-strategy sample --validation-cycles 3
```

### 2. Validation Strategies

**Strategy: `fast`** (for strong correlation)
- Inner loop: F1/MRR only
- RAGAS validation: Only at final cycle
- Time savings: ~75%
- Token savings: ~75%

**Strategy: `adaptive`** (for moderate correlation)
- Inner loop: F1/MRR only
- RAGAS validation: Every 2-3 cycles
- Time savings: ~50%
- Token savings: ~50%

**Strategy: `sample`** (for weak correlation)
- Inner loop: F1/MRR + quick RAGAS (1 run)
- Full RAGAS: Every cycle
- Time savings: ~25%
- Token savings: ~30%

### 3. Pattern-Specific Configuration

Create `config/pattern_validation_strategies.yaml`:

```yaml
# Auto-generated from correlation analysis
validation_strategies:
"""

    # Add pattern-specific recommendations
    for detail in pattern_details:
        pattern_id = detail['pattern_id']
        r_f1 = detail['r_f1']

        if r_f1 > 0.7:
            strategy = "fast"
        elif r_f1 > 0.4:
            strategy = "adaptive"
        else:
            strategy = "sample"

        report += f"  {pattern_id}: {strategy}  # r = {r_f1:.3f}\n"

    report += """
```

---

## Next Steps

1. **Collect more data**: Run more optimization cycles to increase sample size
2. **Implement validation strategies**: Add `--validation-strategy` flag to `run_pattern_fix_poc.py`
3. **Monitor prediction accuracy**: Track "false positives" (F1 improved but answer didn't)
4. **Build predictor model**: Train `answer_correctness = f(F1, MRR, context_relevance, ...)`
5. **A/B test**: Run same pattern with different strategies, compare final outcomes

---

## Appendix: Statistical Notes

**Pearson correlation (r)**:
- Measures linear relationship between two variables
- Range: -1 (perfect negative) to +1 (perfect positive)
- r = 0: No linear relationship

**P-value**:
- Probability that observed correlation occurred by chance
- p < 0.05: Statistically significant (likely real correlation)
- p ≥ 0.05: Not significant (could be random)

**Sample size matters**:
- Small N (< 10): High uncertainty, large confidence intervals
- Medium N (10-30): Moderate confidence
- Large N (> 30): High confidence

**Delta correlations** (Δ metrics):
- More important than absolute correlations
- Answers: "When F1 improves, does answer quality improve?"
- Stronger signal for optimization decisions
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"\n📄 Report generated: {output_file}")
    print(f"\n💡 Key insights:")
    if all_f1_answer_corrs:
        mean_r = np.mean(all_f1_answer_corrs)
        print(f"   Mean F1-Answer correlation: {mean_r:.3f}")
        if mean_r > 0.7:
            print("   ✅ STRONG correlation → Can optimize for F1 (fast and cheap!)")
        elif mean_r > 0.4:
            print("   ⚠️  MODERATE correlation → Hybrid approach recommended")
        else:
            print("   ❌ WEAK correlation → Must use RAGAS metrics directly")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze metric correlations from existing evaluation data"
    )
    parser.add_argument("pattern_id", nargs='?', help="Pattern ID to analyze")
    parser.add_argument("--all", action='store_true', help="Analyze all patterns")
    parser.add_argument(
        "--use-evals",
        action='store_true',
        help="Include evaluation result files (not just iteration JSONL)"
    )
    parser.add_argument("--report", type=Path, help="Generate markdown report")

    args = parser.parse_args()

    output_dir = Path('.diagnostics/correlation_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Find all patterns with data
        diagnostics_dir = Path('.diagnostics')

        if not diagnostics_dir.exists():
            print("❌ No .diagnostics directory found")
            print("   Run some pattern fixes first to generate data")
            return

        pattern_dirs = [d for d in diagnostics_dir.iterdir()
                        if d.is_dir() and d.name != 'correlation_analysis']

        all_data = {}

        for pattern_dir in pattern_dirs:
            pattern_id = pattern_dir.name

            print(f"\n{'='*80}")
            print(f"Pattern: {pattern_id}")
            print(f"{'='*80}")

            try:
                df = load_pattern_data(pattern_id, use_evals=args.use_evals)

                if df is None:
                    print("   ❌ No data found")
                    continue

                all_data[pattern_id] = df

                complete = df.dropna(subset=['url_f1', 'answer_correctness'])
                print(f"   Total samples: {len(df)}")
                print(f"   Complete samples (F1 + Answer): {len(complete)}")

                # Calculate correlations
                corrs = calculate_correlations(df)
                if corrs:
                    print("\n   Correlations:")
                    for name, (r, p) in corrs.items():
                        # Color code output
                        if abs(r) > 0.7:
                            strength = "STRONG"
                        elif abs(r) > 0.4:
                            strength = "MODERATE"
                        else:
                            strength = "WEAK"
                        print(f"     {name}: r = {r:+.3f} (p = {p:.3f}) - {strength}")

                # Generate plots
                if len(complete) >= 3:
                    plot_correlations(df, pattern_id, output_dir)
                else:
                    print("   ⚠️  Insufficient data for plots (need ≥3 complete samples)")

            except Exception as e:
                print(f"   ❌ Failed to analyze {pattern_id}: {e}")
                import traceback
                traceback.print_exc()

        # Generate aggregate report
        if args.report and all_data:
            generate_report(all_data, args.report)
        elif all_data:
            # Auto-generate report
            default_report = output_dir / 'correlation_report.md'
            generate_report(all_data, default_report)

    elif args.pattern_id:
        # Single pattern analysis
        print(f"\n{'='*80}")
        print(f"Pattern: {args.pattern_id}")
        print(f"{'='*80}")

        df = load_pattern_data(args.pattern_id, use_evals=args.use_evals)

        if df is None:
            print("❌ No data found for this pattern")
            print("\nSearched for:")
            print(f"  - .diagnostics/{args.pattern_id}/{args.pattern_id}_iterations.jsonl")
            if args.use_evals:
                print(f"  - .diagnostics/{args.pattern_id}/*result*.json")
            return

        complete = df.dropna(subset=['url_f1', 'answer_correctness'])
        print(f"Total samples: {len(df)}")
        print(f"Complete samples (F1 + Answer): {len(complete)}")

        # Show data preview (only columns that exist)
        print("\nData preview:")
        preview_cols = [col for col in ['ticket_id', 'current_url_f1', 'current_answer_correctness', 'url_f1_delta', 'answer_correctness_delta'] if col in df.columns]
        if not preview_cols:
            preview_cols = [col for col in ['url_f1', 'mrr', 'answer_correctness', 'context_relevance', 'faithfulness'] if col in df.columns]
        print(df[preview_cols].head(10))

        # Per-ticket analysis (if available)
        per_ticket_analysis = analyze_per_ticket_correlations(df)
        if per_ticket_analysis:
            print("\n" + "="*80)
            print("PER-TICKET CORRELATION ANALYSIS")
            print("="*80)

            stats = per_ticket_analysis['aggregate_stats']
            print(f"\nAggregate Statistics (across {stats['n_tickets']} tickets):")
            print(f"  Mean correlation:   r = {stats['mean_r']:+.3f}")
            print(f"  Median correlation: r = {stats['median_r']:+.3f}")
            print(f"  Std deviation:      σ = {stats['std_r']:.3f}")
            print(f"  Range:              {stats['min_r']:+.3f} to {stats['max_r']:+.3f}")
            print(f"\nCorrelation Strength Distribution:")
            print(f"  ✅ Strong   (r > 0.7):   {stats['strong_correlation_tickets']} tickets")
            print(f"  ⚠️  Moderate (r 0.4-0.7): {stats['moderate_correlation_tickets']} tickets")
            print(f"  ❌ Weak     (r < 0.4):   {stats['weak_correlation_tickets']} tickets")

            print(f"\nPer-Ticket Breakdown:")
            print(f"{'Ticket ID':<20} {'Correlation (r)':<18} {'p-value':<12} {'Samples':<10} {'Strength'}")
            print("-" * 80)

            for ticket_id, corr_data in sorted(per_ticket_analysis['per_ticket_correlations'].items()):
                r, p = corr_data['f1_vs_answer']
                n = corr_data['n_samples']

                if abs(r) > 0.7:
                    strength = "✅ STRONG"
                elif abs(r) > 0.4:
                    strength = "⚠️  MODERATE"
                else:
                    strength = "❌ WEAK"

                print(f"{ticket_id:<20} {r:>+7.3f}            {p:>8.4f}    {n:<10} {strength}")

            print("="*80)

        # Calculate overall correlations (pooled across all tickets/iterations)
        corrs = calculate_correlations(df)
        if corrs:
            print("\nOverall Correlations (pooled data):")
            for name, (r, p) in corrs.items():
                if abs(r) > 0.7:
                    strength = "✅ STRONG"
                elif abs(r) > 0.4:
                    strength = "⚠️  MODERATE"
                else:
                    strength = "❌ WEAK"
                print(f"  {name}: r = {r:+.3f} (p = {p:.3f}) - {strength}")

        # Generate plots
        if len(complete) >= 3:
            plot_correlations(df, args.pattern_id, output_dir)
        else:
            print("\n⚠️  Insufficient data for correlation plots (need ≥3 complete samples)")
            print("   Run more optimization iterations to collect more data")

    else:
        parser.print_help()
        print("\n💡 Examples:")
        print("   python scripts/analyze_metric_correlations.py BOOTLOADER_GRUB_ISSUES")
        print("   python scripts/analyze_metric_correlations.py --all --use-evals")
        print("   python scripts/analyze_metric_correlations.py --all --report correlations.md")


if __name__ == '__main__':
    main()
