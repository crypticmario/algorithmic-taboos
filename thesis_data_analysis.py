"""
============================================================
ALGORITHMIC TABOOS - COMPREHENSIVE ANALYSIS SCRIPT
============================================================
Richard Anekwe | University of Limerick | 2026

HOW TO USE:
1. Put all four CSV files in the SAME folder as this script
2. Open your terminal/command prompt
3. Navigate to that folder: cd path/to/folder
4. Install required packages (if not already):
   pip install pandas numpy scipy statsmodels
5. Run: python thesis_analysis.py
6. Results will print to screen AND save to a file called
   'analysis_results.txt' in the same folder

The four CSV files you need:
- global_taboos_2026_20260225_010835_CORRECTED.csv  (Llama 3 8B)
- llama3_1_taboos_2026.csv                          (Llama 3.1)
- mistral_taboos_2026.csv                           (Mistral 7B)
- qwen2_5_taboos_20260324_002612.csv                (Qwen 2.5)
============================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# ============================================================
# STEP 0: Set up output to both screen and file
# ============================================================
class Tee:
    """Write to both screen and file simultaneously."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    def flush(self):
        self.terminal.flush()
        self.file.flush()

sys.stdout = Tee('analysis_results.txt')

print("=" * 80)
print("ALGORITHMIC TABOOS - COMPREHENSIVE ANALYSIS")
print("Richard Anekwe | University of Limerick | 2026")
print("=" * 80)
print()

# ============================================================
# STEP 1: Load all four datasets
# ============================================================
print("STEP 1: Loading datasets...")
print("-" * 40)

# File paths - adjust these if your files are named differently
files = {
    'Llama 3 8B': 'global_taboos_2026_20260225_010835_CORRECTED.csv',
    'Llama 3.1': 'llama3_1_taboos_2026.csv',
    'Mistral 7B': 'mistral_taboos_2026.csv',
    'Qwen 2.5': 'qwen2_5_taboos_20260324_002612.csv'
}

dataframes = {}
for model_name, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"  ERROR: Cannot find {filepath}")
        print(f"  Make sure the file is in: {os.getcwd()}")
        sys.exit(1)
    df = pd.read_csv(filepath)
    print(f"  Loaded {model_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    dataframes[model_name] = df

print()

# ============================================================
# STEP 2: Harmonise column names across all four datasets
# ============================================================
print("STEP 2: Harmonising column names...")
print("-" * 40)

# Each file has slightly different column names.
# Here's what each file actually contains:
#
# Llama 3 8B (CORRECTED):
#   Columns: run_id, identity, passport_tier, prompt_type, time_seconds,
#            hard_refusal, safety_lecture, preamble_length, response
#   NOTE: No 'model' column. hard_refusal and safety_lecture are already correct.
#
# Llama 3.1:
#   Columns: model, run_id, identity, passport_tier, prompt_type, time_seconds,
#            hard_refusal, dialogical_repression, preamble_length, response
#   NOTE: 'hard_refusal' is the refusal metric.
#         'dialogical_repression' is the safety lecture metric (old name).
#
# Mistral 7B:
#   Columns: model, run_id, identity, passport_tier, prompt_type, time_seconds,
#            dialogical_repression, safety_lecture, preamble_length, response
#   NOTE: 'dialogical_repression' is the hard refusal metric (old name).
#         'safety_lecture' is the safety lecture metric.
#
# Qwen 2.5:
#   Columns: model, run_id, identity, passport_tier, prompt_type, time_seconds,
#            dialogical_repression, safety_lecture, preamble_length, response
#   NOTE: Same structure as Mistral.

# --- Llama 3 8B ---
llama3 = dataframes['Llama 3 8B'].copy()
llama3['model'] = 'Llama 3 8B'
# Already has correct column names: hard_refusal, safety_lecture
print(f"  Llama 3 8B: hard_refusal and safety_lecture already correct")

# --- Llama 3.1 ---
llama31 = dataframes['Llama 3.1'].copy()
llama31['model'] = 'Llama 3.1'
# hard_refusal is already there
# 'dialogical_repression' = safety_lecture (old naming)
llama31 = llama31.rename(columns={'dialogical_repression': 'safety_lecture'})
print(f"  Llama 3.1: renamed 'dialogical_repression' -> 'safety_lecture'")

# --- Mistral 7B ---
mistral = dataframes['Mistral 7B'].copy()
mistral['model'] = 'Mistral 7B'
# 'dialogical_repression' = hard_refusal (old naming)
# 'safety_lecture' is already correct
mistral = mistral.rename(columns={'dialogical_repression': 'hard_refusal'})
print(f"  Mistral 7B: renamed 'dialogical_repression' -> 'hard_refusal'")

# --- Qwen 2.5 ---
qwen = dataframes['Qwen 2.5'].copy()
qwen['model'] = 'Qwen 2.5'
# Same structure as Mistral
qwen = qwen.rename(columns={'dialogical_repression': 'hard_refusal'})
print(f"  Qwen 2.5: renamed 'dialogical_repression' -> 'hard_refusal'")

print()

# ============================================================
# STEP 3: Combine into single dataset
# ============================================================
print("STEP 3: Combining into single dataset...")
print("-" * 40)

# Select only the columns we need (in consistent order)
keep_cols = ['model', 'run_id', 'identity', 'passport_tier', 'prompt_type',
             'time_seconds', 'hard_refusal', 'safety_lecture', 'preamble_length', 'response']

for name, df in [('Llama 3 8B', llama3), ('Llama 3.1', llama31),
                 ('Mistral 7B', mistral), ('Qwen 2.5', qwen)]:
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: {name} is missing columns: {missing}")
        for c in missing:
            df[c] = np.nan

all_data = pd.concat([llama3[keep_cols], llama31[keep_cols],
                       mistral[keep_cols], qwen[keep_cols]], ignore_index=True)

# Ensure numeric types
all_data['hard_refusal'] = pd.to_numeric(all_data['hard_refusal'], errors='coerce').fillna(0).astype(int)
all_data['safety_lecture'] = pd.to_numeric(all_data['safety_lecture'], errors='coerce').fillna(0).astype(int)
all_data['preamble_length'] = pd.to_numeric(all_data['preamble_length'], errors='coerce').fillna(0).astype(int)
all_data['time_seconds'] = pd.to_numeric(all_data['time_seconds'], errors='coerce')

# Add Henley scores
henley_scores = {
    'Singaporean': 195, 'Japanese': 193, 'Emirati': 185, 'Irish': 190,
    'Portuguese': 189, 'New Zealander': 188, 'British': 190, 'Canadian': 188,
    'Icelandic': 188, 'American': 189,
    'Nigerian': 46, 'South Sudanese': 29, 'North Korean': 39, 'Eritrean': 38,
    'Somali': 35, 'Pakistani': 33, 'Yemeni': 34, 'Iraqi': 29,
    'Syrian': 27, 'Afghan': 26
}
all_data['henley_score'] = all_data['identity'].map(henley_scores)

print(f"  Combined dataset: {len(all_data)} rows")
print(f"  Models: {sorted(all_data['model'].unique())}")
print(f"  Identities ({all_data['identity'].nunique()}): {sorted(all_data['identity'].unique())}")
print(f"  Prompt types: {sorted(all_data['prompt_type'].unique())}")
print(f"  Passport tiers: {sorted(all_data['passport_tier'].unique())}")

# Save combined dataset
all_data.to_csv('combined_all_models.csv', index=False)
print(f"\n  Saved combined dataset as: combined_all_models.csv")
print()

# ============================================================
# STEP 4: ANALYSIS - Hard Refusal Rates
# ============================================================
print("=" * 80)
print("ANALYSIS 1: HARD REFUSAL RATES BY TIER AND PROMPT TYPE")
print("=" * 80)

for model in ['Llama 3 8B', 'Llama 3.1', 'Mistral 7B', 'Qwen 2.5']:
    mdf = all_data[all_data['model'] == model]
    print(f"\n--- {model} ---")
    print(f"  {'Tier':<12} | {'Positive':>10} | {'Neutral':>10} | {'Negative':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for tier in ['Top 10', 'Bottom 10']:
        rates = []
        for pt in ['positive', 'neutral', 'negative']:
            sub = mdf[(mdf['passport_tier'] == tier) & (mdf['prompt_type'] == pt)]
            rate = sub['hard_refusal'].mean() * 100
            rates.append(f"{rate:>9.1f}%")
        print(f"  {tier:<12} | {rates[0]} | {rates[1]} | {rates[2]}")

print()

# ============================================================
# STEP 5: CHI-SQUARE TESTS
# ============================================================
print("=" * 80)
print("ANALYSIS 2: CHI-SQUARE TESTS (Tier x Hard Refusal, Negative Prompts)")
print("=" * 80)

for model in ['Llama 3 8B', 'Llama 3.1', 'Mistral 7B', 'Qwen 2.5']:
    mdf = all_data[(all_data['model'] == model) & (all_data['prompt_type'] == 'negative')]
    print(f"\n--- {model} ---")

    ct = pd.crosstab(mdf['passport_tier'], mdf['hard_refusal'])
    print(f"  Crosstab:")
    print(f"  {ct.to_string()}")

    if ct.shape == (2, 2):
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        n = len(mdf)
        cramers_v = np.sqrt(chi2 / n)
        print(f"\n  Chi-square(1) = {chi2:.2f}")
        print(f"  p-value = {p:.6f}")
        print(f"  Cramer's V = {cramers_v:.3f}")
        print(f"  n = {n}")

        # Significance stars
        if p < .001:
            sig = "***"
        elif p < .01:
            sig = "**"
        elif p < .05:
            sig = "*"
        else:
            sig = "ns"
        print(f"  Significance: {sig}")
    else:
        print(f"\n  Cannot compute chi-square: no variance in outcome")
        print(f"  (All responses are the same - either all refused or all complied)")

print()

# ============================================================
# STEP 6: COUNTRY-LEVEL REFUSAL (Negative Prompts)
# ============================================================
print("=" * 80)
print("ANALYSIS 3: REFUSAL BY COUNTRY (Negative Prompts)")
print("=" * 80)

for model in ['Llama 3 8B', 'Llama 3.1', 'Mistral 7B', 'Qwen 2.5']:
    mdf = all_data[(all_data['model'] == model) & (all_data['prompt_type'] == 'negative')]
    print(f"\n--- {model} ---")
    by_id = mdf.groupby(['identity', 'passport_tier'])['hard_refusal'].mean() * 100
    by_id = by_id.sort_values(ascending=False)
    for (identity, tier), pct in by_id.items():
        tag = 'TOP' if tier == 'Top 10' else 'BOT'
        marker = ' <<<' if (tag == 'TOP' and pct > 50) else ''  # Flag high-refusal Top 10
        print(f"  [{tag}] {identity:20s}: {pct:6.1f}%{marker}")

print()

# ============================================================
# STEP 7: SAFETY LECTURE RATES (Among Compliant Responses)
# ============================================================
print("=" * 80)
print("ANALYSIS 4: SAFETY LECTURE RATES (Among Compliant Responses Only)")
print("=" * 80)

for model in ['Llama 3 8B', 'Llama 3.1', 'Mistral 7B', 'Qwen 2.5']:
    mdf = all_data[(all_data['model'] == model) & (all_data['hard_refusal'] == 0)]
    print(f"\n--- {model} (n={len(mdf)} compliant responses) ---")
    if len(mdf) == 0:
        print("  No compliant responses to analyse")
        continue
    print(f"  {'Tier':<12} | {'Positive':>10} | {'Neutral':>10} | {'Negative':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for tier in ['Top 10', 'Bottom 10']:
        rates = []
        for pt in ['positive', 'neutral', 'negative']:
            sub = mdf[(mdf['passport_tier'] == tier) & (mdf['prompt_type'] == pt)]
            if len(sub) > 0:
                rate = sub['safety_lecture'].mean() * 100
                rates.append(f"{rate:>8.1f}% ")
            else:
                rates.append(f"{'N/A':>10}")
        print(f"  {tier:<12} | {rates[0]} | {rates[1]} | {rates[2]}")

print()

# ============================================================
# STEP 8: PREAMBLE LENGTH (Among Compliant Responses)
# ============================================================
print("=" * 80)
print("ANALYSIS 5: PREAMBLE LENGTH (Among Compliant Responses)")
print("=" * 80)

for model in ['Llama 3 8B', 'Llama 3.1', 'Mistral 7B', 'Qwen 2.5']:
    mdf = all_data[(all_data['model'] == model) & (all_data['hard_refusal'] == 0)]
    print(f"\n--- {model} ---")
    if len(mdf) == 0:
        print("  No compliant responses")
        continue
    for pt in ['positive', 'neutral', 'negative']:
        sub = mdf[mdf['prompt_type'] == pt]
        top = sub[sub['passport_tier'] == 'Top 10']['preamble_length']
        bot = sub[sub['passport_tier'] == 'Bottom 10']['preamble_length']
        if len(top) > 1 and len(bot) > 1:
            t_stat, p_val = stats.ttest_ind(bot, top)
            pooled_std = np.sqrt((top.std()**2 + bot.std()**2) / 2)
            d = (bot.mean() - top.mean()) / pooled_std if pooled_std > 0 else 0
            print(f"  {pt:8s}:")
            print(f"    Top 10:    M = {top.mean():7.1f}, SD = {top.std():6.1f}, n = {len(top)}")
            print(f"    Bottom 10: M = {bot.mean():7.1f}, SD = {bot.std():6.1f}, n = {len(bot)}")
            print(f"    t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}")
            sig = "***" if p_val < .001 else ("**" if p_val < .01 else ("*" if p_val < .05 else "ns"))
            print(f"    Significance: {sig}")
        else:
            print(f"  {pt:8s}: Insufficient data (Top n={len(top)}, Bot n={len(bot)})")

print()

# ============================================================
# STEP 9: RESPONSE LATENCY
# ============================================================
print("=" * 80)
print("ANALYSIS 6: RESPONSE LATENCY (All Responses)")
print("=" * 80)

for model in ['Llama 3 8B', 'Llama 3.1', 'Mistral 7B', 'Qwen 2.5']:
    mdf = all_data[all_data['model'] == model]
    print(f"\n--- {model} ---")
    for pt in ['positive', 'neutral', 'negative']:
        sub = mdf[mdf['prompt_type'] == pt]
        top = sub[sub['passport_tier'] == 'Top 10']['time_seconds'].dropna()
        bot = sub[sub['passport_tier'] == 'Bottom 10']['time_seconds'].dropna()
        if len(top) > 1 and len(bot) > 1:
            t_stat, p_val = stats.ttest_ind(bot, top)
            pooled_std = np.sqrt((top.std()**2 + bot.std()**2) / 2)
            d = (bot.mean() - top.mean()) / pooled_std if pooled_std > 0 else 0
            print(f"  {pt:8s}:")
            print(f"    Top 10:    M = {top.mean():7.1f}s, SD = {top.std():6.1f}")
            print(f"    Bottom 10: M = {bot.mean():7.1f}s, SD = {bot.std():6.1f}")
            print(f"    t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}")
            sig = "***" if p_val < .001 else ("**" if p_val < .01 else ("*" if p_val < .05 else "ns"))
            print(f"    Significance: {sig}")

print()

# ============================================================
# STEP 10: BINARY LOGISTIC REGRESSION
# ============================================================
print("=" * 80)
print("ANALYSIS 7: BINARY LOGISTIC REGRESSION")
print("(Henley Score predicting Hard Refusal on Negative Prompts)")
print("=" * 80)

# Per-model regressions
for model in ['Llama 3 8B', 'Llama 3.1', 'Mistral 7B', 'Qwen 2.5']:
    mdf = all_data[(all_data['model'] == model) & (all_data['prompt_type'] == 'negative')].copy()
    print(f"\n--- {model} ---")

    if mdf['hard_refusal'].nunique() < 2:
        print("  Cannot fit logistic regression: no variance in outcome")
        print(f"  (All responses are: hard_refusal = {mdf['hard_refusal'].unique()[0]})")
        continue

    try:
        mod = smf.logit("hard_refusal ~ henley_score", data=mdf).fit(disp=0)
        print(f"  Coefficient (B): {mod.params['henley_score']:.4f}")
        print(f"  Odds Ratio (OR): {np.exp(mod.params['henley_score']):.4f}")
        print(f"  p-value: {mod.pvalues['henley_score']:.6f}")
        print(f"  Pseudo R-squared: {mod.prsquared:.4f}")
        sig = "***" if mod.pvalues['henley_score'] < .001 else (
              "**" if mod.pvalues['henley_score'] < .01 else (
              "*" if mod.pvalues['henley_score'] < .05 else "ns"))
        print(f"  Significance: {sig}")
    except Exception as e:
        print(f"  Error fitting model: {e}")

# Combined model (all models together)
print(f"\n--- COMBINED MODEL (all models, negative prompts) ---")
neg_all = all_data[all_data['prompt_type'] == 'negative'].copy()
if neg_all['hard_refusal'].nunique() >= 2:
    try:
        mod = smf.logit("hard_refusal ~ henley_score + C(model)", data=neg_all).fit(disp=0)
        print(f"  Henley coefficient (B): {mod.params['henley_score']:.4f}")
        print(f"  Odds Ratio (OR): {np.exp(mod.params['henley_score']):.4f}")
        print(f"  p-value: {mod.pvalues['henley_score']:.6f}")
        print(f"  Pseudo R-squared: {mod.prsquared:.4f}")
        print(f"\n  Full model summary:")
        print(mod.summary())
    except Exception as e:
        print(f"  Error: {e}")

print()

# ============================================================
# STEP 11: CROSS-MODEL SUMMARY TABLE
# ============================================================
print("=" * 80)
print("SUMMARY TABLE: Cross-Model Comparison")
print("=" * 80)
print()
print(f"{'Model':<15} | {'Neg Ref Top10':>14} | {'Neg Ref Bot10':>14} | {'V (neg)':>8} | {'SL Top10':>10} | {'SL Bot10':>10}")
print("-" * 80)

for model in ['Llama 3 8B', 'Llama 3.1', 'Mistral 7B', 'Qwen 2.5']:
    mdf = all_data[all_data['model'] == model]
    neg = mdf[mdf['prompt_type'] == 'negative']

    top_ref = neg[neg['passport_tier'] == 'Top 10']['hard_refusal'].mean() * 100
    bot_ref = neg[neg['passport_tier'] == 'Bottom 10']['hard_refusal'].mean() * 100

    # Cramer's V
    ct = pd.crosstab(neg['passport_tier'], neg['hard_refusal'])
    if ct.shape == (2, 2):
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / len(neg))
    else:
        v = float('nan')

    # Safety lecture among compliant (all prompt types)
    comp = mdf[mdf['hard_refusal'] == 0]
    top_sl = comp[comp['passport_tier'] == 'Top 10']['safety_lecture'].mean() * 100
    bot_sl = comp[comp['passport_tier'] == 'Bottom 10']['safety_lecture'].mean() * 100

    v_str = f"{v:.3f}" if not np.isnan(v) else "N/A"
    print(f"{model:<15} | {top_ref:>13.1f}% | {bot_ref:>13.1f}% | {v_str:>8} | {top_sl:>9.1f}% | {bot_sl:>9.1f}%")

print()

# ============================================================
# DONE
# ============================================================
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Files created:")
print(f"  1. combined_all_models.csv  - All 2,400 trials in one file")
print(f"  2. analysis_results.txt     - This output saved as text")
print()
print("Next steps:")
print("  - Compare these numbers with the thesis draft")
print("  - Upload analysis_results.txt to your project knowledge")
print("  - Flag any discrepancies for discussion")
print()
print("NOTE ON COLUMN MAPPING:")
print("  - 'dialogical_repression' in Mistral & Qwen files = 'hard_refusal'")
print("  - 'dialogical_repression' in Llama 3.1 file = 'safety_lecture'")
print("  - This reflects the corrected naming convention from the recoding process")
