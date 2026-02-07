"""
Rarity Roulette Pilot Studies: Scoring Rubric & Code
=====================================================
Author: Vera Wilde
Repository: https://github.com/verawilde/rarity-roulette
OSF: [link to your OSF project]

This script scores both pilot studies' test problem responses.

Pilots 1 and 2 used the same 5 test problems but different SoSci Survey
data structures:
  - Pilot 1: Free-text responses in columns TP01s-TP05s (participants typed
    their answer AND showed their work in a single text field)
  - Pilot 2: Structured responses with separate numeric answer fields
    (TP01_01-TP05_01) and Bayesian algorithm coding (BN01-BN03 for 3 of 5
    problems; TP01s-TP05s for show-your-work free text)

Two outcome variables are scored:
  1. ACCURACY: Did the participant give the correct numerical answer?
  2. BAYESIAN ALGORITHM USE: Did the participant use a Bayesian algorithm
     (i.e., identify the correct denominator as true positives + false
     positives)?

Correct answers for each test problem:
  TP01 (Mammography):       5 out of 105   (PPV = 4.76%)
  TP02 (CSAM detection):    8 out of 1007  (PPV = 0.79%)
  TP03 (Prenatal screening): 9 out of 359  (PPV = 2.51%)
  TP04 (Plagiarism):        80 out of 1070 (PPV = 7.48%)
  TP05 (Hate speech):       160 out of 5150 (PPV = 3.11%)
"""

import re
import numpy as np
import pandas as pd


# ============================================================
# CORRECT ANSWERS
# ============================================================

CORRECT_ANSWERS = {
    'TP01': {'numerator': 5, 'denominator': 105, 'ppv_pct': 5/105*100},
    'TP02': {'numerator': 8, 'denominator': 1007, 'ppv_pct': 8/1007*100},
    'TP03': {'numerator': 9, 'denominator': 359, 'ppv_pct': 9/359*100},
    'TP04': {'numerator': 80, 'denominator': 1070, 'ppv_pct': 80/1070*100},
    'TP05': {'numerator': 160, 'denominator': 5150, 'ppv_pct': 160/5150*100},
}


# ============================================================
# ACCURACY SCORING
# ============================================================

def extract_fraction(text):
    """Extract 'X out of Y' or 'X/Y' from free-text response.
    
    Returns (numerator, denominator) or (None, None) if not found.
    """
    if pd.isna(text) or len(str(text).strip()) < 2:
        return None, None
    text = str(text).replace(',', '')  # Remove commas from numbers
    
    # Try "X out of Y" pattern
    match = re.search(r'(\d+)\s*(?:out of|/|:)\s*(\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Try just two numbers
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    
    return None, None


def score_accuracy(text, problem_key):
    """Score accuracy: 1 if correct answer, 0 otherwise.
    
    Accepts exact match on numerator and denominator.
    """
    correct = CORRECT_ANSWERS[problem_key]
    num, denom = extract_fraction(text)
    if num == correct['numerator'] and denom == correct['denominator']:
        return 1
    return 0


def score_accuracy_ppv(text, problem_key, tolerance_pct=0.5):
    """Score accuracy by PPV percentage (alternative scoring).
    
    Accepts answer if PPV is within tolerance_pct of correct value.
    Useful for Pilot 2 where some participants gave percentages.
    """
    correct = CORRECT_ANSWERS[problem_key]
    num, denom = extract_fraction(text)
    if num is not None and denom is not None and denom > 0:
        answer_ppv = num / denom * 100
        if abs(answer_ppv - correct['ppv_pct']) < tolerance_pct:
            return 1
    return 0


# ============================================================
# BAYESIAN ALGORITHM CODING
# ============================================================

def code_bayesian_algorithm(text):
    """Code whether participant used a Bayesian algorithm.
    
    A Bayesian algorithm is defined as: identifying the correct denominator
    as the sum of true positives and false positives (i.e., all positive
    test results), rather than using the total population or some other
    incorrect denominator.
    
    Indicators include:
    - Mentioning "true positives" AND "false positives" together
    - Showing addition of TP + FP to get denominator
    - Mentioning "total positives" or "all flagged/detected"
    - Referencing numerator/denominator structure
    
    Returns 1 (Bayesian) or 0 (non-Bayesian or uncodeable).
    """
    if pd.isna(text) or len(str(text).strip()) < 5:
        return 0
    
    text = str(text).lower()
    
    # Direct indicators of Bayesian reasoning
    bayesian_indicators = [
        'true positive' in text and 'false positive' in text,
        'true positives' in text and 'false positives' in text,
        'correctly' in text and 'incorrectly' in text and ('positive' in text or 'flagged' in text),
        'denominator' in text and ('positive' in text or 'flagged' in text),
        'total' in text and ('positive' in text or 'flagged' in text) and 'population' not in text,
        'all' in text and ('positive' in text or 'flagged' in text) and 'test' in text,
    ]
    
    # Regex patterns for arithmetic showing TP + FP
    pattern_matches = [
        re.search(r'(true|false)\s*(positive|negative).*add', text),
        re.search(r'add(ed|ing)?.*?(true|false|positive)', text),
        re.search(r'\d+\s*\+\s*\d+', text),  # Shows addition like "5 + 100"
        re.search(r'(true positive|tp).*\+.*(false positive|fp)', text),
        re.search(r'plus.*false', text),
    ]
    
    if any(bayesian_indicators) or any(pattern_matches):
        return 1
    return 0


# ============================================================
# PILOT 1 SCORING (free-text only)
# ============================================================

def score_pilot1(df):
    """Score Pilot 1 data.
    
    Expects columns: TP01s-TP05s (free-text show-your-work responses),
    IV01_01 (condition: 1=Control, 2=Treatment), TIME_SUM (seconds).
    """
    for tp_key in CORRECT_ANSWERS:
        col = f'{tp_key}s'  # TP01s, TP02s, etc.
        if col in df.columns:
            df[f'{tp_key}_correct'] = df[col].apply(
                lambda x: score_accuracy(x, tp_key))
            df[f'{tp_key}_bayes'] = df[col].apply(code_bayesian_algorithm)
    
    correct_cols = [f'{tp}_correct' for tp in CORRECT_ANSWERS if f'{tp}_correct' in df.columns]
    bayes_cols = [f'{tp}_bayes' for tp in CORRECT_ANSWERS if f'{tp}_bayes' in df.columns]
    
    df['accuracy'] = df[correct_cols].sum(axis=1)  # out of 5
    df['bayes_proportion'] = df[bayes_cols].mean(axis=1)  # 0 to 1
    df['time_minutes'] = df['TIME_SUM'] / 60
    df['condition'] = df['IV01_01'].map({1: 'Control', 2: 'Treatment'})
    
    return df


# ============================================================
# PILOT 2 SCORING (structured + free-text)
# ============================================================

def score_pilot2(df):
    """Score Pilot 2 data.
    
    Expects columns: TP01_01-TP05_01 (structured answer fields),
    BN01-BN03 (Bayesian algorithm coding for 3 problems),
    TP01s-TP05s (free-text), IV01_01 (condition), TIME_SUM (seconds).
    """
    # Accuracy from structured answer fields
    for tp_key in CORRECT_ANSWERS:
        col = f'{tp_key}_01'
        if col in df.columns:
            df[f'{tp_key}_correct'] = df[col].apply(
                lambda x: score_accuracy(x, tp_key))
    
    # Bayesian algorithm use from BN variables (coded 1=Bayesian)
    # BN01-BN03 cover 3 of 5 problems; for the other 2, use free-text coding
    bayes_cols = []
    for bn_col in ['BN01', 'BN02', 'BN03']:
        if bn_col in df.columns:
            df[f'{bn_col}_bayes'] = (df[bn_col] == 1).astype(int)
            bayes_cols.append(f'{bn_col}_bayes')
    
    # Fall back to free-text coding for remaining problems
    for tp_key in CORRECT_ANSWERS:
        col = f'{tp_key}s'
        if col in df.columns and f'{tp_key}_bayes' not in df.columns:
            df[f'{tp_key}_bayes'] = df[col].apply(code_bayesian_algorithm)
            if f'{tp_key}_bayes' not in bayes_cols:
                bayes_cols.append(f'{tp_key}_bayes')
    
    correct_cols = [f'{tp}_correct' for tp in CORRECT_ANSWERS if f'{tp}_correct' in df.columns]
    
    df['accuracy'] = df[correct_cols].sum(axis=1)
    df['bayes_proportion'] = df[bayes_cols].mean(axis=1) if bayes_cols else np.nan
    df['time_minutes'] = df['TIME_SUM'] / 60
    df['condition'] = df['IV01_01'].map({1: 'Control', 2: 'Treatment'})
    
    # Attention check (Pilot 2 only)
    if 'TP06s' in df.columns:
        df['attn_pass'] = df['TP06s'].str.lower().str.strip() == 'banana'
    
    return df


# ============================================================
# PREREGISTERED EXCLUSION CRITERIA
# ============================================================

def apply_exclusions_pilot2(df):
    """Apply Pilot 2 preregistered exclusion criteria.
    
    Preregistered on OSF:
    1. Failed 1+ attention checks (TP06s != 'banana')
    2. Self-reported non-genuine effort (SC04 == 2 or SC05 == 2)
    3. Self-reported severe distraction
    
    Quality-filtered subsample: time_minutes >= 15 (preregistered)
    
    Note: Pilot 1 did NOT preregister a time cutoff.
    Any time-based filtering on Pilot 1 is post-hoc/exploratory.
    """
    excluded = pd.Series(False, index=df.index)
    
    if 'attn_pass' in df.columns:
        excluded |= ~df['attn_pass']
    if 'SC04' in df.columns:
        excluded |= (df['SC04'] == 2)
    if 'SC05' in df.columns:
        excluded |= (df['SC05'] == 2)
    
    full_sample = df[~excluded].copy()
    quality_sample = full_sample[full_sample['time_minutes'] >= 15].copy()
    
    return full_sample, quality_sample


# ============================================================
# ANALYSIS
# ============================================================

def compute_effect_size(df, dv, label=""):
    """Compute treatment-control difference with Cohen's d and 95% CI."""
    from scipy import stats
    
    control = df[df['condition'] == 'Control'][dv].dropna()
    treatment = df[df['condition'] == 'Treatment'][dv].dropna()
    
    mean_diff = treatment.mean() - control.mean()
    
    pooled_std = np.sqrt(
        ((len(control)-1)*control.std()**2 + (len(treatment)-1)*treatment.std()**2) / 
        (len(control) + len(treatment) - 2)
    )
    
    d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # SE of Cohen's d
    se_d = np.sqrt(
        (len(control) + len(treatment)) / (len(control) * len(treatment)) + 
        d**2 / (2 * (len(control) + len(treatment)))
    )
    
    df_approx = len(control) + len(treatment) - 2
    t_crit = stats.t.ppf(0.975, df_approx)
    d_ci_lo = d - t_crit * se_d
    d_ci_hi = d + t_crit * se_d
    
    print(f"\n{label}")
    print(f"  Control: M={control.mean():.3f}, SD={control.std():.3f}, n={len(control)}")
    print(f"  Treatment: M={treatment.mean():.3f}, SD={treatment.std():.3f}, n={len(treatment)}")
    print(f"  Difference: {mean_diff:+.3f}")
    print(f"  Cohen's d: {d:+.3f} (95% CI: {d_ci_lo:+.3f} to {d_ci_hi:+.3f})")
    
    return {'d': d, 'ci_lo': d_ci_lo, 'ci_hi': d_ci_hi, 
            'mean_diff': mean_diff, 'n_control': len(control), 'n_treatment': len(treatment)}


if __name__ == '__main__':
    print("Rarity Roulette Scoring Rubric")
    print("=" * 60)
    print("\nCorrect answers:")
    for tp, vals in CORRECT_ANSWERS.items():
        print(f"  {tp}: {vals['numerator']} out of {vals['denominator']} "
              f"(PPV = {vals['ppv_pct']:.2f}%)")
    print("\nTo use: import this module and call score_pilot1() or score_pilot2()")
    print("on your SoSci Survey dataframe.")
