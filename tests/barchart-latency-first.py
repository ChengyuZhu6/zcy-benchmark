import matplotlib.pyplot as plt
import numpy as np

# Data
instances = ["8", "16", "32", "64"]
average_constant_input_latency = [1.526516702, 2.025441562, 3.787614267, 8.915414104]
average_limited_input_latency = [2.072720504, 2.750537927, 7.560184518, 14.71684159]
average_unlimited_input_latency = [2.956530185, 9.477127556, 23.68571466, 59.11739473]

# Set the positions and width for the bars
positions = np.arange(len(instances))
bar_width = 0.25

# Plotting
plt.bar(positions, average_constant_input_latency, bar_width, label='Constant Input')
plt.bar(positions + bar_width, average_limited_input_latency, bar_width, label='Limited Input')
plt.bar(positions + 2*bar_width, average_unlimited_input_latency, bar_width, label='Unlimited Input')

# Adding labels
plt.xlabel('Concurrency')
plt.ylabel('Latency (s)')
plt.title('First token Latency Comparison')
plt.xticks(positions + bar_width, instances)

# Adding legend
plt.legend()

plt.savefig('first_latency_with_different_input_tokens.png')

# Display the bar chart
plt.show()
