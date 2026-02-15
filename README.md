# ML-MAD: Machine Learning Models for Accelerated Discovery

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-mad.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**ML-MAD** (Machine Learning Models for Accelerated Discovery) is a state-of-the-art Streamlit application designed for atomistic simulations of molecules and materials using universal machine learning interatomic potentials (MLIPs). 

This platform allows researchers and students to easily test, compare, and benchmark foundational MLIPs without writing complex code.

## 🚀 Key Features

- **Multiple Input Methods:** Upload structure files (XYZ, CIF, POSCAR, etc.), select from predefined examples (Water, Methane, Caffeine, etc.), or paste file content directly.
- **Advanced 3D Visualization:** Interactive 3D visualization of molecular and crystal structures using `py3Dmol`.
- **Comprehensive Calculation Tasks:**
  - Single point Energy Calculation
  - Energy + Forces Calculation
  - Geometry Optimization (Atomic positions)
  - Cell + Geometry Optimization (Full relaxation)
- **Built-in Optimizers:** Supports BFGS, LBFGS, and FIRE optimizers via ASE.
- **Result Export:** Download optimized structures in XYZ format.

## 🧠 Supported Foundation Models

ML-MAD integrates several state-of-the-art universal MLIPs:

- **MACE:** Multiple versions including MPA, OMAT, MATPES (r2SCAN/PBE), and MP (Small, Medium, Large).
- **FairChem (Meta):** UMA Small and ESEN (MD/SM) models.

## 💻 Local Installation

To run ML-MAD locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ML-MAD.git
   cd ML-MAD
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-ml.txt
   ```
   *Note: ML packages may require specific hardware (GPU) or additional setup for optimal performance.*

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment to Streamlit Cloud

When deploying to Streamlit Cloud, ensure you handle the following:

1. **Choose dependency scope:**
   - Streamlit Cloud installs only `requirements.txt` by default.
   - `requirements.txt` already includes `requirements-ml.txt`, so core MACE/FairChem calculations are enabled by default.
2. **Hugging Face Token:** Some models require access to Hugging Face. Add your `HF_TOKEN` to the Streamlit Secrets:
   ```toml
   # .streamlit/secrets.toml
   [HF_TOKEN]
   token = "your_huggingface_token_here"
   ```
3. **Resource Limits:** Streamlit Cloud has memory and CPU/GPU limitations. For large systems or intensive optimizations, running locally is recommended. The app includes a built-in atom limit (500 atoms) when running in a cloud environment to ensure stability.

## 🛠 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Atomic Simulations:** [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/)
- **Visualization:** [py3Dmol](https://3dmol.csb.pitt.edu/)
- **Deep Learning:** PyTorch, MACE, FairChem.

## 📄 License

The code in this repository is provided for research and educational purposes. Please note that individual models (like MACE OMAT or FairChem) may be subject to their own licenses (e.g., ASL or Meta's Acceptable Use Policy). Refer to the sidebar in the app for specific license warnings.

## 🤝 Acknowledgments

Made by [Sebin Devasia](https://sebindevasiamx.wixsite.com/sebin).

Special thanks to the developers of MACE and FairChem for providing these incredible foundational models to the scientific community.