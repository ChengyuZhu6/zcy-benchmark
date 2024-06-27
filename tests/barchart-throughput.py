import matplotlib.pyplot as plt
import numpy as np

# Data
instances = ["8", "16", "32", "64"]
throughput_constant_input_locust = [84.480, 158.720, 269.653, 378.880]
throughput_limited_input_locust = [68.693, 122.027, 179.200, 235.520]
throughput_unlimited_input_locust = [66.987, 95.573, 98.987, 75.093]

# Set the positions and width for the bars
positions = np.arange(len(instances))
bar_width = 0.25

# Plotting
plt.bar(positions, throughput_constant_input_locust, bar_width, label='Constant Input')
plt.bar(positions + bar_width, throughput_limited_input_locust, bar_width, label='Limited Input')
plt.bar(positions + 2*bar_width, throughput_unlimited_input_locust, bar_width, label='Unlimited Input')

# Adding labels
plt.xlabel('Instances')
plt.ylabel('Throughput (token/s)')
plt.title('Throughput Comparison')
plt.xticks(positions + bar_width, instances)

# Adding legend
plt.legend()

plt.savefig('throughput_with_different_input_tokens.png')

# Display the bar chart
plt.show()
