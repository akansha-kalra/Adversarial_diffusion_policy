__author__ = "akansha_kalra"
import numpy as np
import matplotlib.pyplot as plt

def get_mean(x):
    return np.round(np.mean(x),decimals=2)
def get_Std(x):
    return np.round(np.std(x),decimals=2)
dp_noattack=[0.955,1,0.909]

dp_targeted=[0.7,0.76,0.68]
mean_dp_no_attack=get_mean(dp_noattack)
mean_dp_targeted_attack=get_mean(dp_targeted)

print(f"DP-Unattacked performance {mean_dp_no_attack} and targeted attack performance {mean_dp_targeted_attack}")

lstm_noattack=[0.591,0.727,0.727]
lstm_targeted=[0.26,0.12,0.46]

mean_lstm_no_attack=get_mean(lstm_noattack)
mean_lstm_targeted=get_mean(lstm_targeted)
print(f"LSTM-Unattacked performance {mean_lstm_no_attack} and targeted attack performance {mean_lstm_targeted}")

# Data for the graph
algorithms = ["Vanilla BC", "LSTM-GMM", "IBC", "Diffusion Policy-C", "VQ-BET"]
algorithms = ["Vanilla BC", "LSTM-GMM", "IBC", "Diffusion Policy-C"]
mean_iou_no_attack = [0.01, mean_lstm_no_attack, 0.01 , mean_dp_no_attack]
mean_iou_targeted_attack = [0.01, mean_lstm_targeted, 0.01, mean_dp_targeted_attack]
# mean_iou_untargeted_attack = []

std_targeted_attack=[0.00,get_Std(lstm_targeted),0.00,get_Std(dp_targeted)]
std_no_attack=[0.00,get_Std(lstm_noattack),0.00,get_Std(dp_noattack)]
#
# algorithms = ["LSTM-GMM", "Diffusion Policy-C"]
# mean_iou_no_attack = [mean_lstm_no_attack,  mean_dp_no_attack]
# mean_iou_targeted_attack = [mean_lstm_targeted, mean_dp_targeted_attack]
#
#
# std_targeted_attack=[get_Std(lstm_targeted) , get_Std(dp_targeted)]
# std_no_attack=[get_Std(lstm_noattack),get_Std(dp_noattack)]
x = np.arange(len(algorithms))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the bars
bars1 = ax.bar(x - width, mean_iou_no_attack, width, label='No Attack', color='green', yerr=std_no_attack, capsize=5, ecolor='black')
bars2 = ax.bar(x, mean_iou_targeted_attack, width, yerr=std_targeted_attack, label='Targeted Attack', color='red', capsize=5, ecolor='black')
# bars3 = ax.bar(x + width, mean_iou_untargeted_attack, width, label='Untargeted Attack', color='blue')

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Algorithm')
ax.set_ylabel('Normal Success Rate')
# ax.set_title('Success Rate Under Different Attack Types')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.set_ylim([0, 1])
ax.legend(title='Attack Type')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='lightgray')

# Adding data labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height==0.01:
            height_write=0.00
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height_write:.2f}',
                    ha='center', va='bottom', fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)

add_labels(bars1)
add_labels(bars2)
# add_labels(bars3)

plt.tight_layout()
plt.savefig("NoTitle_Fixed_tool_hang_UAP")
plt.show()
