import matplotlib.pyplot as plt

# Data
instances = [
    "16/64",
    "24/96",
    "32/128",
    "48/196",
    "64/256",
    "94/384"
]

throughput = [
    7.250,
    8.754,
    9.883,
    9.984,
    9.499,
    9.621
]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(instances, throughput, color='skyblue')

plt.xlabel('Instance (vCPUs/Memory (GiB))')
plt.ylabel('e2e throughput (token/s)')
plt.title('Throughput on the single instance with batch size 1')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('singe_instance_kserve+torchserve+ipex.png')

# Display the bar chart
plt.show()
