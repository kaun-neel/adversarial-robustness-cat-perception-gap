import json
import matplotlib.pyplot as plt

with open('adversarial_results.json', 'r') as f:
    data = json.load(f)
epsilons = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1]
success_counts = {eps: 0 for eps in epsilons}
total_images = len(data)

#Calculate the Success Rate 
for img, results in data.items():
    for eps in epsilons:
        if results[str(eps)]["success"]:
            success_counts[eps] += 1

success_rates = [(success_counts[eps] / total_images) * 100 for eps in epsilons]

plt.figure(figsize=(10, 6))
plt.plot(epsilons, success_rates, marker='o', linewidth=3, markersize=8, color='#d32f2f', label='I-FGSM Attack')
plt.title('Targeted Adversarial Attack Success Rate\n(Class: Cat $\\rightarrow$ Rug)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Perturbation Magnitude (Epsilon)', fontsize=12)
plt.ylabel('Attack Success Rate (%)', fontsize=12)
plt.ylim(-5, 105) # Keep the Y-axis clean from 0 to 100
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvspan(0.005, 0.02, color='#5c6bc0', alpha=0.2, label='The Perception Gap')

plt.legend(loc='lower right', fontsize=12)
plt.savefig('perception_gap_figure.png', dpi=300, bbox_inches='tight')
print("Graph generated! Check your folder for 'perception_gap_figure.png'.")