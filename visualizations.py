"""
NBA Career Longevity Predictor - Visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("nba_career_longevity_data.csv")

# Create output folder for plots
os.makedirs("plots", exist_ok=True)

# Style settings
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

# Distribution of remaining competitive seasons
fig, ax = plt.subplots()
df['REMAINING_COMPETITIVE_SEASONS'].hist(bins=range(0, 20), edgecolor='black', ax=ax)
ax.set_xlabel("Remaining Competitive Seasons")
ax.set_ylabel("Count (Player-Seasons)")
ax.set_title("Distribution of Remaining Competitive Seasons")
plt.tight_layout()
plt.savefig("plots/01_target_distribution.png")
plt.close()
print("Saved: 01_target_distribution.png")

# Years in league vs remaining competitive seasons
fig, ax = plt.subplots()
grouped = df.groupby('YEARS_IN_LEAGUE')['REMAINING_COMPETITIVE_SEASONS'].mean()
ax.bar(grouped.index, grouped.values, color='steelblue', edgecolor='black')
ax.set_xlabel("Years in League")
ax.set_ylabel("Avg Remaining Competitive Seasons")
ax.set_title("Average Remaining Competitive Seasons by Years in League")
plt.tight_layout()
plt.savefig("plots/02_years_in_league_vs_remaining.png")
plt.close()
print("Saved: 02_years_in_league_vs_remaining.png")

# Cumulative minutes vs remaining competitive seasons
fig, ax = plt.subplots()
sample = df[df['IS_COMPETITIVE']].sample(n=min(3000, len(df)), random_state=42)
scatter = ax.scatter(
    sample['CUMULATIVE_MINUTES'],
    sample['REMAINING_COMPETITIVE_SEASONS'],
    alpha=0.3, s=10, c='steelblue'
)
ax.set_xlabel("Cumulative Career Minutes")
ax.set_ylabel("Remaining Competitive Seasons")
ax.set_title("Cumulative Minutes Played vs. Remaining Competitive Seasons")
plt.tight_layout()
plt.savefig("plots/03_cumulative_minutes_vs_remaining.png")
plt.close()
print("Saved: 03_cumulative_minutes_vs_remaining.png")

# PIE + PPG distributions - competitive vs non-competitive
has_pie = 'PIE' in df.columns

fig, axes = plt.subplots(1, 2 if has_pie else 1, figsize=(14 if has_pie else 7, 6))
if not has_pie:
    axes = [axes]

df.boxplot(column='PPG', by='IS_COMPETITIVE', ax=axes[0])
axes[0].set_xlabel("Is Competitive Season")
axes[0].set_ylabel("Points Per Game")
axes[0].set_title("PPG: Competitive vs Non-Competitive")

if has_pie:
    df.boxplot(column='PIE', by='IS_COMPETITIVE', ax=axes[1])
    axes[1].set_xlabel("Is Competitive Season")
    axes[1].set_ylabel("Player Impact Estimate (PIE)")
    axes[1].set_title("PIE: Competitive vs Non-Competitive")

plt.suptitle("Validating the Competitive Season Label", y=1.02)
plt.tight_layout()
plt.savefig("plots/04_competitive_label_validation.png")
plt.close()
print("Saved: 04_competitive_label_validation.png")

# Distance from peak PPG vs remaining seasons
fig, ax = plt.subplots()
competitive = df[df['IS_COMPETITIVE']].copy()
competitive['PEAK_DIST_BIN'] = pd.cut(
    competitive['DISTANCE_FROM_PEAK'],
    bins=[0, 2, 5, 10, 15, 30],
    labels=['0-2', '2-5', '5-10', '10-15', '15+']
)
binned = competitive.groupby('PEAK_DIST_BIN')['REMAINING_COMPETITIVE_SEASONS'].mean()
ax.bar(range(len(binned)), binned.values, color='coral', edgecolor='black')
ax.set_xticks(range(len(binned)))
ax.set_xticklabels(binned.index)
ax.set_xlabel("Distance from Career Peak PPG")
ax.set_ylabel("Avg Remaining Competitive Seasons")
ax.set_title("Distance from Peak Performance vs. Remaining Competitive Seasons")
plt.tight_layout()
plt.savefig("plots/05_peak_distance_vs_remaining.png")
plt.close()
print("Saved: 05_peak_distance_vs_remaining.png")

# Correlation heatmap of key features
fig, ax = plt.subplots(figsize=(10, 8))
feature_cols = [
    'YEARS_IN_LEAGUE', 'MPG', 'PPG', 'RPG', 'APG', 'PIE',
    'CUMULATIVE_MINUTES', 'CUMULATIVE_GAMES',
    'DISTANCE_FROM_PEAK', 'PPG_CHANGE', 'MPG_CHANGE',
    'REMAINING_COMPETITIVE_SEASONS'
]
feature_cols = [c for c in feature_cols if c in df.columns]
corr_df = df[feature_cols].dropna().corr()
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png")
plt.close()
print("Saved: 06_correlation_heatmap.png")

# Summary 
print("\n" + "=" * 50)
print("SUMMARY STATS")
print("=" * 50)
print(f"Total player-season records: {len(df)}")
print(f"Unique players: {df['PLAYER_ID'].nunique()}")
print(f"Competitive seasons: {df['IS_COMPETITIVE'].sum()} ({df['IS_COMPETITIVE'].mean()*100:.1f}%)")
print(f"\nTarget variable (remaining competitive seasons):")
print(df['REMAINING_COMPETITIVE_SEASONS'].describe().to_string())
print(f"\nFeature correlations with target:")
target_corr = corr_df['REMAINING_COMPETITIVE_SEASONS'].drop('REMAINING_COMPETITIVE_SEASONS').sort_values()
print(target_corr.to_string())
