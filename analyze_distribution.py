import pandas as pd

# Load the World Series data
df = pd.read_csv('world_series_winners.csv')

print("=" * 80)
print("WORLD SERIES WINNERS DISTRIBUTION ANALYSIS (1903-2024)")
print("=" * 80)
print(f"\nTotal World Series Championships: {len(df)}")
print(f"Years analyzed: 1903-2024 (no series in 1904 and 1994)")

# Count wins by team
wins_by_team = df['Winner'].value_counts().sort_values(ascending=False)

print(f"\nUnique teams that have won: {len(wins_by_team)}")

# Calculate percentages
total_championships = len(df)
win_percentages = (wins_by_team / total_championships * 100).round(2)

# Create a detailed DataFrame with counts and percentages
distribution = pd.DataFrame({
    'Team': wins_by_team.index,
    'Championships': wins_by_team.values,
    'Percentage': win_percentages.values
})

print("\n" + "=" * 80)
print("CHAMPIONSHIP DISTRIBUTION BY TEAM")
print("=" * 80)

# Display full distribution
for idx, row in distribution.iterrows():
    team_name = row['Team']
    championships = int(row['Championships'])
    percentage = row['Percentage']

    # Create a simple bar visualization
    bar_length = int(percentage / 2)  # Scale down for display
    bar = '█' * bar_length

    print(f"{team_name:30s} | {championships:3d} wins | {percentage:6.2f}% | {bar}")

print("\n" + "=" * 80)
print("TOP 10 TEAMS BY CHAMPIONSHIPS")
print("=" * 80)

top_10 = distribution.head(10)
for idx, row in top_10.iterrows():
    print(f"{idx+1:2d}. {row['Team']:30s} - {int(row['Championships']):3d} championships ({row['Percentage']:5.2f}%)")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Calculate some interesting stats
top_3_percentage = distribution.head(3)['Percentage'].sum()
top_5_percentage = distribution.head(5)['Percentage'].sum()
top_10_percentage = distribution.head(10)['Percentage'].sum()

print(f"Top 3 teams control:  {top_3_percentage:.2f}% of all championships")
print(f"Top 5 teams control:  {top_5_percentage:.2f}% of all championships")
print(f"Top 10 teams control: {top_10_percentage:.2f}% of all championships")

# Save the distribution to CSV
distribution.to_csv('world_series_distribution.csv', index=False)
print(f"\nDetailed distribution saved to: world_series_distribution.csv")
