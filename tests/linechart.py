import matplotlib.pyplot as plt

# Data
throughput_constant_input_locust = [84.480, 158.720, 269.653, 378.880]
average_constant_input_latency = [81.08469688, 83.15333393, 86.84894963, 92.01766773]
throughput_limited_input_locust = [68.693, 122.027, 179.200, 235.520]
average_limited_input_latency = [81.26264443, 87.95812098, 88.79762473, 108.1337258]
throughput_unlimited_input_locust = [66.987, 95.573, 98.987, 75.093]
average_unlimited_input_latency = [87.0132095, 79.88720981, 106.3743889, 223.173909]

labels = ["8,1", "16,2", "32,4", "64,8"]

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot average latency on the left vertical axis
average_latency_lines1, = ax1.plot(throughput_constant_input_locust, average_constant_input_latency, 'b-o',  label='Average Latency with constant input tokens')
average_latency_lines2, = ax1.plot(throughput_limited_input_locust, average_limited_input_latency, 'y-o',  label='Average Latency with limited input tokens')
average_latency_lines3, = ax1.plot(throughput_unlimited_input_locust, average_unlimited_input_latency, 'g-o', label='Average Latency with unlimited input tokens')

ax1.set_xlabel('Throughput (tokens / second)')
ax1.set_ylabel('Average latency (ms / token)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Add labels to data points
for i, label in enumerate(labels):
    ax1.text(throughput_constant_input_locust[i], average_constant_input_latency[i], label)
for i, label in enumerate(labels):
    ax1.text(throughput_limited_input_locust[i], average_limited_input_latency[i], label)
for i, label in enumerate(labels):
    ax1.text(throughput_unlimited_input_locust[i], average_unlimited_input_latency[i], label)

average_token_sla_lines = ax1.axhline(y=100, color='red', linestyle='--')
ax1.text(throughput_constant_input_locust[1], 100, 'Average latency SLA(100 ms)', color='red', va='top', ha='right')
ax1.text(-0.05, 100, '', color='red', va='center', ha='right', transform=ax1.get_yaxis_transform())
ax1.legend(loc='upper right')

# Show the grid
ax1.grid(True)

table_data = [
    ["Cluster", "4x2s SPR"],
    ["Instance", "48vcpus, 192GiB Memory"],
    ["Core binding", "NRI with topology-aware"],
    # ["input length", "1024"],
    ["output length", "128"],
]

# Add the table below the plot
table = plt.table(cellText=table_data, colLabels=None, colLoc='center', cellLoc='center', loc='bottom', bbox=[0.1, -0.5, 0.8, 0.3])

# Adjust layout to make space for the table
plt.subplots_adjust(bottom=0.3)

plt.savefig('throughput_vs_latency_with_different_input_tokens.png')

# Display the plot
plt.show()
