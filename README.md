Self-Pruning Neural Network for Efficient Deep Learning

Overview
This project implements a Self-Pruning Neural Network that automatically removes less important neurons during training using sparsity regularization.
The objective is to reduce model complexity while maintaining good accuracy.

Key Concept
A gating mechanism is introduced for each neuron.
During training:

* Important neurons remain active
* Less important neurons are suppressed (pruned)
This enables automatic model optimization.

Methodology
1. Train a neural network on CIFAR-10 dataset
2. Apply sparsity regularization using lambda (λ)
3. Learn gate values for each neuron
4. Evaluate:
    * Accuracy
    * Sparsity
    * Trade-off between them

Result
| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|------------|-------------------|--------------|
| 0.0001     | 71.66             | 0.00         |
| 0.001      | 62.86             | 51.41        |
| 0.01       | 50.83             | 0.00         |

Observations
* Lambda = 0.0001 gives highest accuracy
* Lambda = 0.001 produces highest sparsity (~51%)
* Lambda = 0.01 causes performance degradation

There is a clear trade-off between accuracy and sparsity.

Visualizations
The project includes:
* Gate value distribution plots
* Training loss curves
* Accuracy vs sparsity trade-off
* Per-layer sparsity analysis

All results are available in the results folder.

How to Run
Install dependencies:
pip install -r requirements.txt

Run the model:
python Self_pruning_network.py

Project Structure
Self_pruning_network.py   → Main implementation
results/                 → Output graphs and results
.gitignore               → Excluded files

Conclusion
Self-pruning helps reduce model size without heavy manual tuning.
A moderate lambda value provides the best balance between accuracy and sparsity.

Future Work
* Apply pruning to deeper architectures
* Improve pruning strategy
* Optimize inference speed

Author
Vishal R (RA2311056010028)

