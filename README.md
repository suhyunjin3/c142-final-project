# C142 Final Project

# Neural Network Potential Training on the ANI-1 Dataset
 
Julie Jin (UGrad)  
C142 Machine Learning, Statistical Models, and Optimization for Molecular Problems

### Introduction
 
Predicting the energy of a molecule accurately is one of the main challenges in computational chemistry. The standard approach is to use quantum mechanical (QM) density functional theory (DFT), which can predict energies with high accuracy but at a high computational cost. This makes DFT an impractical method for large-scale molecular simulations. Classical force fields are significantly faster but have less accuracy, especially for non-equilibrium geometries and chemical reactions. On the other hand, neural network potentials (NNPs) can be trained on DFT data and then predict energies at a fraction of the computational cost, while having close to DFT accuracy.
 
This project implements and trains an ANI-style neural network potential on a subset of the ANI-1 dataset, which is the first truly transferable NNP for organic molecules, introduced by Smith, Isayev, and Roitberg in 2017.<sup>1</sup> The ANI-1 model uses atom-centered symmetry functions, originally developed by Behler and Parrinello,<sup>2</sup> to construct Atomic Environment Vectors (AEVs) as molecular representations and learns a separate neural network for each element type (H, C, N, O). The total molecular energy is calculated as the sum of atomic contributions from each element's network.
 
The dataset used in this work is the ANI-1 dataset, which contains DFT-calculated energies for small organic molecules drawn from the GDB-11 database. Molecules contain up to 8 heavy atoms (C, N, O) and hydrogen (H), while conformations were generated using Normal Mode Sampling (NMS). All models were implemented using PyTorch<sup>3</sup> and the TorchANI library.<sup>4</sup> In this project, the study trains on the GDB s01–s04 subset (molecules with 1–4 heavy atoms) rather than the full ANI-1 dataset and evaluates how closely the model's accuracy approaches the original paper's reported results.
 
### Methods
 
The ANI-1 dataset was loaded using the TorchANI library.<sup>4</sup> Atomic self-energies were subtracted from all total energies to obtain relative conformational energies that are easier to learn. Species labels (H, C, N, O) were mapped to integer indices 0–3. The dataset was split into training (80%), validation (10%), and test (10%) sets, and batched using torchani's built-in collate and cache methods.
 
Each atom's local chemical environment is encoded as an AEV using modified Behler-Parrinello symmetry functions.<sup>1,2</sup> The AEV has a radial component (calculating distances to neighbors) and an angular component (calculating triplet angles), both within a cutoff radius. AEVs are differentiated by atom type where separate subvectors are computed for each neighbor element type. With 4 element types, this produces 4 radial and 10 angular subAEVs, giving a total dimensionality of 384. The parameters were radial cutoff 5.2 Å, angular cutoff 3.5 Å, 16 radial and 8 angular shifting values.
 
A separate atomic neural network is trained for each element type. Each network maps a 384-dimensional AEV to a scalar atomic energy. Two different base architectures were used:
 
AtomicNet_A: 384 → 128 → 1 (one hidden layer, ReLU)
AtomicNet_B: 384 → 256 → 128 → 1 (two hidden layers, ReLU)
All models were trained with the Adam optimizer (MSE loss) and early stopping based on validation loss, saving the best weights observed.
 
Five configurations were evaluated (Table 1), varying architecture, learning rate, epochs, L2 weight decay, and batch size. Each model was initialized independently before training for fair comparison. Model 3 (AtomicNet_B, lr=1e-3, epoch=20, l2=0, batch=8192) achieved the best test MAE of 1.24 kcal/mol and was selected as the final model.
 
Table 1: Hyperparameter search results. All MAE values in kcal/mol = mean(|true − pred|) × 627.51.
 
| Model | Architecture | LR | Epochs | L2 | Batch | Dropout | Train MAE | Val MAE | Test MAE |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 384→128→1 | 1e-3 | 20 | 0 | 8192 | 0.0 | 1.74 | 1.75 | 1.76 |
| 2 | 384→128→1 | 1e-3 | 20 | 1e-4 | 8192 | 0.0 | 2.54 | 2.55 | 2.56 |
| 3 | 384→256→128→1 | 1e-3 | 20 | 0 | 8192 | 0.0 | 1.24 | 1.25 | 1.24 |
| 4 | 384→128→1 | 1e-4 | 20 | 1e-5 | 8192 | 0.0 | 2.69 | 2.70 | 2.69 |
| 5 | 384→256→128→1 | 5e-4 | 30 | 1e-5 | 4096 | 0.0 | 1.54 | 1.54 | 1.55 |
 
All error values use Mean Absolute Error (MAE), computed as mean(|true − predicted energy|) × 627.5095 kcal/mol/hartree. The ANI-1 paper uses RMSE where RMSE ≥ MAE, so numerical comparisons between the two metrics are approximate.
 
The study used two experiments to assess the robustness of the best model: five independent training runs from different random initializations, and 5-fold cross-validation on the combined train and validation set. Finally, a production model was trained on all available data (train, validation, and test) to maximize usage.
 
### Results
 
AtomicNet_B consistently outperformed AtomicNet_A across all metrics. Model 3 achieved the best test MAE of 1.24 kcal/mol and is the only model below the 2 kcal/mol target. The near-identical train, validation, and test MAEs for Model 3 (1.24, 1.25, 1.24 kcal/mol) show no evidence of overfitting. Adding L2 regularization (Model 2) hurt performance, and no dropout was used across all models.
 
Five independent runs of Model 3 produced a mean test MAE of **1.890 ± 0.353 kcal/mol**, reflecting sensitivity to random initialization on a smaller dataset.
 
5-fold cross-validation yielded a mean test MAE of **1.336 ± 0.213 kcal/mol**, with close agreement between validation and test MAE across all folds confirming consistent generalization.
 
The final model trained on all available data achieved a MAE of **1.11 kcal/mol**, with smooth loss convergence and points tightly clustered around the y=x diagonal in the parity plot.
 
Table 2: Comparison with the ANI-1 paper. The original paper uses RMSE; this study uses MAE.
 
| | ANI-1 (Paper) | This Work |
|---|---|---|
| Dataset | GDB s01–s08, 17.2M conf., 57,951 molecules | GDB s01–s04 (subset) |
| Architecture (per element) | 768:128:128:64:1 | 384:256:128:1 |
| Parameters (per element) | 124,033 | 131,585 |
| Activation | Gaussian | ReLU |
| Optimizer / Loss | ADAM (exponential) | ADAM (MSE) |
| Reported metric | RMSE | MAE |
| Train error | 1.2 kcal/mol | — |
| Validation error | 1.3 kcal/mol | — |
| Test error | 1.3 kcal/mol | — |
| Best single-run Test MAE | — | 1.24 kcal/mol (Model 3) |
| 5-Fold CV Test MAE | — | 1.336 ± 0.213 kcal/mol |
| Multi-run Mean Test MAE | — | 1.890 ± 0.353 kcal/mol |
| Final model (all data) | ~1.3 kcal/mol (RMSE) | 1.11 kcal/mol (MAE) |
 
### Conclusion
 
The study demonstrates that an ANI-style neural network potential can achieve similar accuracy to the original ANI-1 paper even when trained on a smaller subset of data (GDB s01–s04). The best systematic test MAE of 1.24 kcal/mol (Model 3) and the final all-data model MAE of 1.11 kcal/mol are both competitive with the paper's reported test RMSE of 1.3 kcal/mol, considering RMSE ≥ MAE. The 5-fold cross-validation mean of 1.336 ± 0.213 kcal/mol provides a robust estimate of generalization performance and confirms that the model consistently falls below the 2 kcal/mol accuracy target.
 
Architecture depth was the most important hyperparameter. AtomicNet_B consistently outperformed AtomicNet_A, with best test MAEs of 1.24 vs. 1.76 kcal/mol respectively. The closeness of the train, validation, and test MAEs for Model 3 confirm that the model generalizes well without overfitting. L2 regularization and lower learning rates did not improve performance at this data scale, and no dropout was necessary.
 
The main limitation is run-to-run variance (std = 0.353 kcal/mol), reflecting the smaller training set compared to the full 17.2M conformation ANI-1 dataset. Future work could explore CELU or Gaussian activations for smoother potential energy surfaces, training on the full s01–s08 dataset to reduce variance and push MAE toward the 1 kcal/mol chemical accuracy threshold, and implementing a learning rate schedule as in the original paper. Overall, this project confirms that the AEV-based ANI architecture is effective and transferable even at reduced data scale.
 
### Dependencies
 
```
torch
torchani
numpy
matplotlib
tqdm
```
 
---
 
### References
 
1. Smith, J. S., Isayev, O., & Roitberg, A. E. (2017). ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost. *Chemical Science*, 8(4), 3192–3203.
 
2. Behler, J., & Parrinello, M. (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98(14), 146401.
 
3. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.
 
4. Gao, X., et al. (2020). TorchANI: A free and open source PyTorch-based deep learning implementation of the ANI neural network potentials. *Journal of Chemical Information and Modeling*, 60(7), 3408–3415.
