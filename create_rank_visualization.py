import matplotlib.pyplot as plt
import numpy as np

# Our actual data
categories = ['Determiners\n(the, a)', 'Prepositions\n(on, with)', 'Verbs\n(fell, rolled)', 'Objects\n(ball)', 'Objects\n(homework)']
ranks = [0, 0, 12.5, 32, 11988]

# Create bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green', 'green', 'yellow', 'orange', 'red']
bars = ax.bar(categories, ranks, color=colors)

# Log scale for visibility
ax.set_yscale('symlog')
ax.set_ylabel('Prediction Rank (log scale)', fontsize=12)
ax.set_title('Positional Encoding Bias in BERT: Function vs Content Words', fontsize=14)

# Add value labels
for bar, rank in zip(bars, ranks):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(rank)}', ha='center', va='bottom')

ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Good prediction threshold')
plt.tight_layout()
plt.savefig('positional_encoding_ranks.png', dpi=150)
print("Saved to positional_encoding_ranks.png")
