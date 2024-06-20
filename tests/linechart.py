import matplotlib.pyplot as plt

# Data
throughput = [84.480, 158.720, 269.653, 340.480, 378.880]
latency = [81.08469688, 87.08187613, 104.6528675, 126.038419, 149.4743464]
labels = ["concurrency=8", "concurrency=16", "concurrency=32", "concurrency=48", "concurrency=64"]

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(throughput, latency, marker='o')

# Titles and labels
plt.title('llama2-7b-bf16 on kserve+torchserve+ipex')
plt.xlabel('Throughput (tokens / second)')
plt.ylabel('Latency (average ms / token)')

# Add labels to data points
for i, label in enumerate(labels):
    plt.text(throughput[i], latency[i], label)

# Add a red line at y = 120 labeled as 'SLA'
plt.axhline(y=120, color='red', linestyle='--', label='SLA')

# Show the grid
plt.grid(True)
plt.legend()

table_data = [
    ["Cluster", "4x2s SPR"],
    ["Instance", "48vcpus, 192GiB Memory"],
    ["Core bundle", "NRI with topology-aware"],
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
