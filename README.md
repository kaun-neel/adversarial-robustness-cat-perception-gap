
# Adversarial Robustness and the Cat-to-Object Perception Gap in Deep Learning Models

[![DOI](https://zenodo.org/badge/1207143752.svg)](https://doi.org/10.5281/zenodo.19499543)

## Overview

This project investigates adversarial robustness in deep learning models, with a specific focus on the **cat-to-object perception gap** — a phenomenon where models misclassify cats as generic or unrelated objects under adversarial perturbations.

The study explores how subtle input modifications can significantly degrade model performance and reveal inconsistencies in learned visual representations.

---

## Objectives

* Analyze robustness of deep learning models against adversarial attacks
* Identify patterns in misclassification (cat → object shift)
* Quantify performance degradation under perturbations
* Visualize perception gaps using experimental data

---

## Key Contributions

*  Empirical analysis of adversarial vulnerabilities in image classification
*  Identification of the **cat-to-object perception gap** phenomenon
*  Visualization of misclassification trends and robustness breakdown
*  Experimental pipeline for adversarial evaluation

---

## ⚙️ Methodology

1. **Dataset Preparation**

   * Curated dataset focusing on cat images and object classes

2. **Model Evaluation**

   * Tested standard deep learning models for classification

3. **Adversarial Attacks**

   * Applied perturbation techniques to evaluate robustness

4. **Analysis**

   * Compared predictions before and after perturbations
   * Generated visual and statistical insights

---

## 📊 Results

* Significant drop in classification accuracy under adversarial conditions
* Frequent misclassification of cats into unrelated object categories
* Clear evidence of weak feature generalization in models

---

## 📁 Repository Structure

```
.
├── cat_dataset/                # Dataset used for experiments
├── main.py                    # Main execution script
├── generate_charts.py         # Visualization and analysis
├── adversarial_results.json   # Experiment results
├── perception_gap.png         # Key visualization
├── paper.pdf                  # Research paper
├── README.md
└── .gitignore
```

---

## 📄 Research Paper

📥 [Download Paper](./Cat_Adversarial.pdf)

---

## 📚 Citation

```
Bose, Indraneel (2026).
Adversarial Robustness and the Cat-to-Object Perception Gap in Deep Learning Models.
Zenodo. https://doi.org/10.5281/zenodo.19499544
```

---

## Tech Stack

* Python
* NumPy, Pandas
* Matplotlib / Seaborn
* Deep Learning Frameworks (TensorFlow / PyTorch)

---

## Future Work

* Extend analysis to other animal/object classes
* Evaluate defense mechanisms against adversarial attacks
* Improve model robustness using adversarial training

---

## Author

**Indraneel Bose**
Computer Science & Data Analytics
IIT Patna

---

## ⭐ Acknowledgment

This project was developed as an independent research effort exploring robustness challenges in modern AI systems.
