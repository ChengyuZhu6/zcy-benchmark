import matplotlib.pyplot as plt
import numpy as np

# Data
instances = ["8", "16", "32", "64"]
average_constant_input_latency = [81.08469688, 83.15333393, 86.84894963, 92.01766773]
average_limited_input_latency = [81.26264443, 87.95812098, 88.79762473, 108.1337258]
average_unlimited_input_latency = [87.0132095, 79.88720981, 106.3743889, 223.173909]

# Set the positions and width for the bars
positions = np.arange(len(instances))
bar_width = 0.25

# Plotting
plt.bar(positions, average_constant_input_latency, bar_width, label='Constant Input')
plt.bar(positions + bar_width, average_limited_input_latency, bar_width, label='Limited Input')
plt.bar(positions + 2*bar_width, average_unlimited_input_latency, bar_width, label='Unlimited Input')

# Adding labels
plt.xlabel('Concurrency')
plt.ylabel('Latency (ms)')
plt.title('Latency Comparison')
plt.xticks(positions + bar_width, instances)

# Adding legend
plt.legend()

plt.savefig('latency_with_different_input_tokens.png')

# Display the bar chart
plt.show()
