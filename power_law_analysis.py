import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Load the data
df = pd.read_csv('world_series_winners.csv')
wins_by_team = df['Winner'].value_counts().sort_values(ascending=False)

print("=" * 80)
print("POWER LAW DISTRIBUTION ANALYSIS")
print("=" * 80)

# Prepare data for analysis
wins = wins_by_team.values
ranks = np.arange(1, len(wins) + 1)

print(f"\nData Summary:")
print(f"Number of teams: {len(wins)}")
print(f"Total championships: {wins.sum()}")
print(f"Max wins (Yankees): {wins[0]}")
print(f"Min wins: {wins[-1]}")
print(f"Mean wins per team: {wins.mean():.2f}")
print(f"Median wins per team: {np.median(wins):.2f}")

# 1. VISUAL TEST: Log-log plot
plt.figure(figsize=(12, 10))

# Plot 1: Regular distribution
plt.subplot(2, 2, 1)
plt.bar(ranks, wins, color='steelblue', alpha=0.7)
plt.xlabel('Rank (by championships)', fontsize=10)
plt.ylabel('Number of Championships', fontsize=10)
plt.title('World Series Championships by Rank', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 2: Log-log plot (key test for power law)
plt.subplot(2, 2, 2)
plt.loglog(ranks, wins, 'o', markersize=8, color='darkred', alpha=0.7, label='Actual data')

# Fit power law: wins = a * rank^(-b)
def power_law(x, a, b):
    return a * x**(-b)

# Fit the power law
popt, pcov = curve_fit(power_law, ranks, wins)
a_fit, b_fit = popt

# Generate fitted line
x_fit = np.linspace(ranks.min(), ranks.max(), 100)
y_fit = power_law(x_fit, a_fit, b_fit)

plt.loglog(x_fit, y_fit, '-', linewidth=2, color='orange',
           label=f'Power law fit: y = {a_fit:.2f} × x^{-b_fit:.2f}')
plt.xlabel('Log(Rank)', fontsize=10)
plt.ylabel('Log(Championships)', fontsize=10)
plt.title('Log-Log Plot (Power Law Test)', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

print(f"\n" + "=" * 80)
print("POWER LAW FIT RESULTS")
print("=" * 80)
print(f"Power law equation: Championships = {a_fit:.2f} × Rank^(-{b_fit:.2f})")
print(f"Exponent (α): {b_fit:.3f}")

# Calculate R² for power law fit
residuals = wins - power_law(ranks, a_fit, b_fit)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((wins - np.mean(wins))**2)
r_squared_power = 1 - (ss_res / ss_tot)
print(f"R² for power law fit: {r_squared_power:.4f}")

# Plot 3: Semi-log plot (for exponential comparison)
plt.subplot(2, 2, 3)
plt.semilogy(ranks, wins, 'o', markersize=8, color='darkgreen', alpha=0.7, label='Actual data')

# Fit exponential: wins = a * exp(-b * rank)
def exponential(x, a, b):
    return a * np.exp(-b * x)

try:
    popt_exp, _ = curve_fit(exponential, ranks, wins, maxfev=10000)
    a_exp, b_exp = popt_exp
    y_exp = exponential(x_fit, a_exp, b_exp)
    plt.semilogy(x_fit, y_exp, '-', linewidth=2, color='purple',
                 label=f'Exponential fit: y = {a_exp:.2f} × e^(-{b_exp:.3f}x)')

    # Calculate R² for exponential fit
    residuals_exp = wins - exponential(ranks, a_exp, b_exp)
    ss_res_exp = np.sum(residuals_exp**2)
    r_squared_exp = 1 - (ss_res_exp / ss_tot)

    print(f"\nExponential fit: Championships = {a_exp:.2f} × e^(-{b_exp:.3f} × Rank)")
    print(f"R² for exponential fit: {r_squared_exp:.4f}")
except:
    print("\nExponential fit failed (data may not be exponential)")
    r_squared_exp = 0

plt.xlabel('Rank', fontsize=10)
plt.ylabel('Log(Championships)', fontsize=10)
plt.title('Semi-Log Plot (Exponential Test)', fontsize=12, fontweight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# Plot 4: Residuals plot
plt.subplot(2, 2, 4)
residuals_power = wins - power_law(ranks, a_fit, b_fit)
plt.scatter(ranks, residuals_power, color='darkred', alpha=0.6, s=50)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Rank', fontsize=10)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=10)
plt.title('Power Law Fit Residuals', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('power_law_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: power_law_analysis.png")

# Statistical tests
print(f"\n" + "=" * 80)
print("DISTRIBUTION CHARACTERISTICS")
print("=" * 80)

# Gini coefficient (inequality measure)
def gini_coefficient(x):
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n

gini = gini_coefficient(wins)
print(f"Gini coefficient: {gini:.4f}")
print(f"  (0 = perfect equality, 1 = perfect inequality)")
print(f"  Value of {gini:.4f} indicates {'HIGH' if gini > 0.5 else 'MODERATE' if gini > 0.3 else 'LOW'} inequality")

# Concentration ratio (% held by top k teams)
top_5_pct = wins[:5].sum() / wins.sum() * 100
top_10_pct = wins[:10].sum() / wins.sum() * 100
print(f"\nConcentration ratios:")
print(f"  Top 5 teams hold: {top_5_pct:.2f}% of championships")
print(f"  Top 10 teams hold: {top_10_pct:.2f}% of championships")

# Kolmogorov-Smirnov test comparing to fitted power law
predicted = power_law(ranks, a_fit, b_fit)
ks_stat, ks_pvalue = stats.ks_2samp(wins, predicted)
print(f"\nKolmogorov-Smirnov test:")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pvalue:.4f}")

print(f"\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Interpret results
print(f"\n1. Power Law Evidence:")
if b_fit > 0.5 and b_fit < 3.0:
    print(f"   ✓ Exponent α = {b_fit:.3f} is in typical range (0.5-3.0) for power laws")
else:
    print(f"   ✗ Exponent α = {b_fit:.3f} is outside typical range (0.5-3.0)")

if r_squared_power > 0.8:
    print(f"   ✓ R² = {r_squared_power:.4f} indicates excellent fit")
elif r_squared_power > 0.6:
    print(f"   ≈ R² = {r_squared_power:.4f} indicates moderate fit")
else:
    print(f"   ✗ R² = {r_squared_power:.4f} indicates poor fit")

print(f"\n2. Model Comparison:")
if r_squared_power > r_squared_exp + 0.05:
    print(f"   Power law (R²={r_squared_power:.4f}) fits better than exponential (R²={r_squared_exp:.4f})")
elif r_squared_exp > r_squared_power + 0.05:
    print(f"   Exponential (R²={r_squared_exp:.4f}) fits better than power law (R²={r_squared_power:.4f})")
else:
    print(f"   Power law and exponential fits are similar (both R² ≈ {r_squared_power:.4f})")

print(f"\n3. Overall Assessment:")
if r_squared_power > 0.7 and 0.5 < b_fit < 3.0 and gini > 0.4:
    print(f"   ✓ Distribution shows STRONG power law characteristics")
    print(f"     - Linear on log-log scale (R² = {r_squared_power:.4f})")
    print(f"     - High inequality (Gini = {gini:.4f})")
    print(f"     - Heavy tail (few teams dominate)")
elif r_squared_power > 0.5:
    print(f"   ≈ Distribution shows MODERATE power law characteristics")
    print(f"     - Some linearity on log-log scale (R² = {r_squared_power:.4f})")
    print(f"     - Limited sample size may affect fit")
else:
    print(f"   ✗ Distribution does NOT closely follow a power law")
    print(f"     - Poor fit on log-log scale (R² = {r_squared_power:.4f})")

print(f"\n4. Interpretation:")
print(f"   This distribution reflects the cumulative advantage effect:")
print(f"   - Successful teams attract better players and more resources")
print(f"   - This creates a 'rich get richer' dynamic")
print(f"   - Common in competitive systems with feedback loops")

print("\n" + "=" * 80)
