import matplotlib.pyplot as plt

# Data
throughput = [84.480, 158.720, 269.653, 309.333, 340.480, 378.880, 413.867, 445.440]
average_latency = [81.08469688, 83.15333393, 86.84894963, 88.57769472, 92.47948183, 92.01766773, 100.7087318, 104.8033153]
first_token_latency = [1.526516702, 2.025441562, 3.787614267, 4.890155458, 5.546423982, 8.915414104, 10.84772262, 13.14988339]
labels = ["concurrency=8(BS=1)", "concurrency=16(BS=2)", "concurrency=32(BS=4)", "concurrency=40(BS=5)", "concurrency=48(BS=6)", "concurrency=64(BS=8)", "concurrency=80(BS=10)", "concurrency=96(BS=12)"]

# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot average latency on the left vertical axis
average_latency_lines, = ax1.plot(throughput, average_latency, 'b-o', label='Average Latency (ms/token)')
ax1.set_xlabel('Throughput (tokens / second)')
ax1.set_ylabel('Average latency (ms / token)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Add labels to data points
for i, label in enumerate(labels):
    ax1.text(throughput[i], average_latency[i], label)

average_token_sla_lines = ax1.axhline(y=100, color='red', linestyle='--')
ax1.text(throughput[1], 100, 'Average latency SLA(100 ms)', color='red', va='top', ha='right')
ax1.text(-0.05, 100, '', color='red', va='center', ha='right', transform=ax1.get_yaxis_transform())
ax1.legend(loc='upper left')

# Create a second y-axis for the first token latency
ax2 = ax1.twinx()
first_token_latency_lines, = ax2.plot(throughput, first_token_latency, 'g-s', label='First Token Latency (s)')
ax2.set_ylabel('First Token Latency (s)', color='g')
ax2.tick_params(axis='y', labelcolor='g')
for i, label in enumerate(labels):
    ax2.text(throughput[i], first_token_latency[i], label)
first_token_sla_lines = ax2.axhline(y=3, color='orange', linestyle='--')
ax2.text(throughput[-1], 3, 'First token latency SLA(3s)', color='orange', va='bottom', ha='right')
first_token_sla_lines_2 = ax2.axhline(y=4, color='orange', linestyle='--')
ax2.text(throughput[-1], 4, 'First token latency SLA(4s)', color='orange', va='bottom', ha='right')

ax2.text(1.017, 3, '3', transform=ax2.get_yaxis_transform(), color='orange', va='center', ha='right')

lines = [average_latency_lines, first_token_latency_lines, average_token_sla_lines, first_token_sla_lines]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')

# Show the grid
ax1.grid(True)

table_data = [
    ["Cluster", "4x2s SPR"],
    ["Instance", "48vcpus, 192GiB Memory"],
    ["Core binding", "NRI with topology-aware"],
    ["input length", "1024"],
    ["output length", "128"],
]

# Add the table below the plot
table = plt.table(cellText=table_data, colLabels=None, colLoc='center', cellLoc='center', loc='bottom', bbox=[0.1, -0.5, 0.8, 0.3])

# Adjust layout to make space for the table
plt.subplots_adjust(bottom=0.3)

plt.savefig('kserve+torchserve+ipex.png')

# Display the plot
plt.show()
