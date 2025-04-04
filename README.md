# Vision Transformer for CMS HCAL Data Quality Monitoring Classification

## Overview

This repository contains the solution for **Test 1 (Vision Transformer Based Classification)** of the ML4DQM evaluation test, submitted as part of the Google Summer of Code (GSoC) 2025 application process for the ML4SCI organization.

The primary goal of this test was to develop a **Vision Transformer (ViT)** model capable of classifying synthetic CMS Hadronic Calorimeter (HCAL) DigiOccupancy maps based on their originating dataset run (representing different operational conditions or periods).

## Context

* **Program:** Google Summer of Code 2025
* **Organization:** ML4SCI
* **Project:** ML4DQM: Machine Learning for Data Quality Monitoring of the CMS Detector
* **Task:** Evaluation Test 1

## Dataset

The task utilized two synthetic datasets provided via CERNBox, representing HCAL DigiOccupancy (hit multiplicity):

* **Source Files:** `Run357479_Dataset_iodic.npy`, `Run355456_Dataset_jqkne.npy` (as identified in the final working environment).
* **Format:** NumPy arrays (`float64`) with shape `(10000, 64, 72)`, representing (Lumi-sections/Samples, iEta Index, iPhi Index).
* **Characteristics:** The data is known to be sparse, containing a significant fraction of zero entries (~79.8% observed). Non-zero values represent hit counts in detector cells.

## Methodology / Approach

The solution was developed using Python and TensorFlow/Keras, following these main steps:

### 1. Exploratory Data Analysis (EDA)
* Initial inspection confirmed shapes, data types, and sparsity.
* Value distributions (histograms of non-zero values) were analyzed.
* Average spatial occupancy maps (mean across samples for each run) were calculated and analyzed statistically, revealing distinct hotspot locations and intensity profiles.
* Statistics for corresponding individual sample slices were compared, confirming instance-level differences consistent with average patterns.

### 2. Data Preprocessing
* The two datasets were loaded and concatenated (total 20,000 samples).
* Binary labels were assigned (0 for Run 357479, 1 for Run 355456).
* Data was reshaped to `(N, 64, 72, 1)` to add a channel dimension suitable for vision models.
* The dataset was split into stratified Train (60% / 12,000 samples), Validation (20% / 4,000 samples), and Test (20% / 4,000 samples) sets.
* Min-Max normalization was applied to scale data to the [0, 1] range. The scaler was fitted *only* on the training data and then applied to all three sets.
* Data arrays were converted to `float32` and labels to `int32`.

### 3. Model Architecture (ViT)
* A standard Vision Transformer was implemented using the TensorFlow/Keras functional API with custom layers for patching and encoding.
* **Key Hyperparameters:**
    * Patch Size: `(8, 9)` (resulting in 64 patches)
    * Projection (Embedding) Dimension: 128
    * Transformer Layers: 4
    * Attention Heads: 4
    * MLP Dimension (in Transformer Block): 256
    * Classifier Head MLP Units: `[64]`
    * Final Activation: Sigmoid
* Total Parameters: ~556k

### 4. Training
* TensorFlow's `tf.distribute.MirroredStrategy` was used for efficient training on available GPUs (tested on dual T4).
* Optimized input pipelines were created using `tf.data.Dataset` (caching, shuffling, batching, prefetching).
* The model was compiled with the AdamW optimizer (lr=3e-4), Binary Crossentropy loss, and metrics (Accuracy, AUC, PRC).
* `ModelCheckpoint` and `EarlyStopping` callbacks were used, monitoring `val_auc`.
* Training converged very quickly, achieving perfect validation metrics by Epoch 3 and stopping early via the callback.

### 5. Evaluation
* The best saved model (`vit_model_checkpoint.keras`) was loaded using the `custom_objects` argument to handle the custom `Patches` and `PatchEncoder` layers.
* The model was evaluated on the held-out test set.


## Results

The Vision Transformer model achieved **perfect classification performance** on the unseen test set, demonstrating its ability to effectively learn the differences between the two synthetic datasets:

* **Test Loss:** 0.0000
* **Test Accuracy:** 1.0000
* **Test AUC (ROC):** 1.0000
* **Test PRC (AUC):** 1.0000

The Receiver Operating Characteristic (ROC) curve visually confirms this perfect separation:

![Test Set ROC Curve](roc_curve_test_set.png "Test Set ROC Curve (AUC = 1.0000)")


## Conclusion

The implemented Vision Transformer model successfully learned the distinguishing spatial features within the synthetic HCAL DigiOccupancy data, resulting in perfect classification accuracy between the two datasets on unseen test data. This outcome validates the potential of ViTs for analyzing spatial detector patterns in the context of Data Quality Monitoring, although the rapid convergence and perfect score suggest the synthetic datasets used in this test were highly separable.
