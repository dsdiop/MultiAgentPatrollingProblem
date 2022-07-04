import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

baseline = pd.read_csv('./run-Greedy_baseline_no_networked-tag-train_accumulated_reward.csv')
baseline['Scenario'] = 'Without network restrictions'
networked = pd.read_csv('./run-Greedy_networked-tag-train_accumulated_reward.csv')
networked['Scenario'] = 'Network restricted'

df = pd.concat((baseline, networked), ignore_index=True)


sns.set_style('darkgrid')
g = sns.lineplot(data = df.groupby('Scenario').rolling(window=100, min_periods = 1, center = False).mean(), x = 'Step', y = 'Value', hue='Scenario', linewidth=3)
g.set_xlabel('Episode')
g.set_ylabel('Mean Accumulated Reward')
g.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
g.fontsize = 9
plt.tight_layout()
plt.show()