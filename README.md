# 🧠 Regularizing Brain Age Prediction via Gated Knowledge Distillation

This repository implements **Brain Age Prediction** using **Gated Knowledge Distillation (GKD)** regularization, based on PyTorch.

The method is trained and evaluated on four public brain imaging datasets:

- [IXI Dataset](http://brain-development.org/)
- [OASIS-3](https://www.oasis-brains.org/)
- [ADNI](https://ida.loni.usc.edu/)
- [1000 Functional Connectomes Project (FCP)](http://www.nitrc.org/projects/fcon_1000)

---

## 🛠️ Setup

### 🔄 Preprocessing

All MRI images should be preprocessed and z-score normalized.

> 🔧 **Preprocessing code is in preparation.**  
> In the meantime, we recommend using [intensity-normalization](https://github.com/jcreinhold/intensity-normalization) for intensity normalization.

---

### 📦 Requirements

Install the following dependencies:

```bash
pip install torch torchvision nibabel numpy scikit-learn transformations logging

