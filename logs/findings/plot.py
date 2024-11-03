import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib as mpl

# Set Seaborn style for better aesthetics
# sns.set(style="whitegrid")  # Options include "darkgrid", "white", "whitegrid", "ticks"
sns.set_palette("muted")  # Use a muted color palette suitable for publication

# Set font to Times New Roman
# mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'sans-serif'
mpl.rcParams['font.size'] = 20  # Set default font size for readability
# mpl.rcParams['font.weight'] = 'bold'  # Make all fonts bold
# mpl.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
# mpl.rcParams['axes.titleweight'] = 'bold'  # Bold titles
mpl.rcParams['xtick.labelsize'] = 14  # Set x-tick font size
mpl.rcParams['ytick.labelsize'] = 14  # Set y-tick font size
mpl.rcParams['xtick.color'] = 'black'  # Ensure x-tick color is strong
mpl.rcParams['ytick.color'] = 'black'  # Ensure y-tick color is strong
mpl.rcParams['axes.titlesize'] = 16  # Title font size, if applicable

# Load the data for all six files
file_paths = [
    'blind_craftsmen_7_target_rew_per_step_productMDP_TD3_experience_generation_episode_rew.csv',
    'blind_craftsman_rew_per_step_env_config_7_cont_experience_generation_episode_rew.csv',
    'gold_mine_7_target_rew_per_step_productMDP_TD3_experience_generation_episode_rew.csv',
    'gold_mine_rew_per_step_env_config_7_cont_experience_generation_episode_rew.csv',
    'dungeon_quest_7_target_rew_per_step_productMDP_TD3_experience_generation_episode_rew.csv',
    'dungeon_quest_rew_per_step_env_config_7_cont_experience_generation_episode_rew.csv'
]

data = [pd.read_csv(path) for path in file_paths]

# Define smoothing function
def smooth(values, smoothing=0.992):
    smoothed_values = []
    last = values[0]
    for value in values:
        smoothed_value = last * smoothing + (1 - smoothing) * value
        smoothed_values.append(smoothed_value)
        last = smoothed_value
    return smoothed_values

# Extract and smooth data for each file
smoothed_data = []
for d in data:
    steps = d['step']
    rewards = d['experience_generation/episode_rew']
    smoothed_rewards = smooth(rewards, smoothing=0.997)
    steps_filtered_1m = steps[steps <= 2000000]
    smoothed_rewards_filtered_1m = smoothed_rewards[:len(steps_filtered_1m)]
    smoothed_data.append((steps_filtered_1m, smoothed_rewards_filtered_1m))

# Create figure with GridSpec layout
fig = plt.figure(figsize=(10.8, 8.8), dpi=600)
gs = gridspec.GridSpec(2, 2)  # 2 rows, 2 columns

def set_plot_style(ax, spine_width=1.5, tick_labelsize=20):
    # Bold border
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    # Set y-axis limits and specific tick marks
    ax.set_ylim(-200, 100)
    ax.set_yticks([-200, -150, -100, -50, 0, 50, 100])
    # Set font size for x-ticks and y-ticks
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

# First comparison 
ax1 = fig.add_subplot(gs[0, 0])
steps_filtered_2m_1 = smoothed_data[0][0]
rewards_filtered_2m_1 = smoothed_data[0][1]
ax1.plot(steps_filtered_2m_1, rewards_filtered_2m_1, label="Dyn Distill", color="#1f77b4", linewidth=1.5, antialiased=True)

steps_filtered_2m_2 = smoothed_data[1][0]
rewards_filtered_2m_2 = smoothed_data[1][1]
ax1.plot(steps_filtered_2m_2, rewards_filtered_2m_2, label="Vanilla TD3", color="#9467bd", linewidth=1.5, antialiased=True)

ax1.set_xlabel("Step", fontsize=20)
ax1.legend(loc="lower right", fontsize=20)
set_plot_style(ax1)
# ax1.set_title("(e)", fontsize=20, loc="left")

# Second comparison 
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
steps_filtered_2m_3 = smoothed_data[2][0]
rewards_filtered_2m_3 = smoothed_data[2][1]
ax2.plot(steps_filtered_2m_3, rewards_filtered_2m_3, color="#1f77b4", linewidth=1.5, antialiased=True)

steps_filtered_2m_4 = smoothed_data[3][0]
rewards_filtered_2m_4 = smoothed_data[3][1]
ax2.plot(steps_filtered_2m_4, rewards_filtered_2m_4, color="#9467bd", linewidth=1.5, antialiased=True)

ax2.set_xlabel("Step", fontsize=20)
set_plot_style(ax2)
# ax2.set_title("(f)", fontsize=20, loc="left")
ax2.yaxis.set_visible(False)

# Third comparison
ax3 = fig.add_subplot(gs[1, 0])
steps_filtered_1m = smoothed_data[4][0][smoothed_data[4][0] <= 1000000]
rewards_filtered_1m = smoothed_data[4][1][:len(steps_filtered_1m)]
ax3.plot(steps_filtered_1m, rewards_filtered_1m, color="#1f77b4", linewidth=1.5, antialiased=True)

steps_filtered_1m_2 = smoothed_data[5][0][smoothed_data[5][0] <= 1000000]
rewards_filtered_1m_2 = smoothed_data[5][1][:len(steps_filtered_1m_2)]
ax3.plot(steps_filtered_1m_2, rewards_filtered_1m_2, color="#9467bd", linewidth=1.5, antialiased=True)

ax3.set_xlabel("Step", fontsize=20)
set_plot_style(ax3)
# ax3.set_title("(g)", fontsize=20, loc="left")

# Adjust layout and save as PNG and SVG
plt.tight_layout()
output_path = './img/all_cont.png'
svg_path = './img/all_cont.svg'
plt.savefig(output_path, format='png', dpi=600)
plt.savefig(svg_path, format='svg', dpi=600)
plt.show()
