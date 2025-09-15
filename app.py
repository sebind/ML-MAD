import streamlit as st
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.visualize import view
import py3Dmol
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import pandas as pd
from orb_models.forcefield import pretrained

# ================================
# 1. SETUP STREAMLIT PAGE
# ================================
st.set_page_config(page_title="MLIP Atomistic Simulation App", layout="wide")

st.title("⚛️ MLIP Atomistic Simulation App")
st.markdown("""
This app allows you to perform atomistic simulations using pre-trained foundational MLIPs such as **FairChem** and **ORB**.  
Upload a structure file (CIF/XYZ), choose a model, and run energy/force calculations or relaxations.
""")

# ================================
# 2. MODEL HELPERS
# ================================
@st.cache_resource
def get_fairchem_model(selected_model, model_path, device, task_type):
    mlip = pretrained_mlip(selected_model, model_path=model_path, device=device, task=task_type)
    return FAIRChemCalculator(mlip)

@st.cache_resource
def get_orb_model(model_name, device):
    return pretrained.calculator(model_name, device=device)

# ================================
# 3. SIDEBAR SETTINGS
# ================================
st.sidebar.header("⚙️ Simulation Settings")

model_type = st.sidebar.radio("Select Model Type:", ["FairChem", "ORB"])
device = st.sidebar.radio("Select Device:", ["cpu", "cuda"])
selected_task_type = st.sidebar.radio("Select Task Type:", ["energy", "forces", "stress"])

if model_type == "FairChem":
    selected_model = st.sidebar.selectbox("Choose FairChem Model:", ["foundation_model"])
    model_path = st.sidebar.text_input("Model Path (Optional)", "")
elif model_type == "ORB":
    selected_model = st.sidebar.selectbox("Choose ORB Model:", ["orb-v2", "orb-v1"])

run_calc = st.sidebar.button("Run Calculation")

# ================================
# 4. UPLOAD STRUCTURE
# ================================
uploaded_file = st.file_uploader("📂 Upload CIF/XYZ file", type=["cif", "xyz"])

atoms = None
if uploaded_file:
    atoms = read(uploaded_file)
    st.success("File uploaded successfully!")
    
    # Show 3D visualization
    st.subheader("🧩 Structure Visualization")
    xyz_str = atoms.get_positions()
    mol = py3Dmol.view(width=400, height=400)
    mol.addModel(atoms.get_chemical_symbols().__str__(), "xyz")
    mol.setStyle({"stick": {}})
    mol.zoomTo()
    mol.show()

# ================================
# 5. RUN CALCULATION
# ================================
if run_calc and atoms:
    st.subheader("🚀 Running Calculation...")

    if model_type == "FairChem":
        st.write("Setting up FairChem calculator...")
        calc = get_fairchem_model(selected_model, model_path, device, selected_task_type)

    elif model_type == "ORB":
        st.write("Setting up ORB calculator...")
        calc = get_orb_model(selected_model, device)

    atoms.set_calculator(calc)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    st.write(f"**Total Energy:** {energy:.6f} eV")
    st.write("**Forces (eV/Å):**")
    st.dataframe(pd.DataFrame(forces, columns=["Fx", "Fy", "Fz"]))

    # Relaxation option
    relax = st.checkbox("🔄 Perform Relaxation")
    if relax:
        st.write("Running relaxation with BFGS...")
        dyn = BFGS(atoms)
        dyn.run(fmax=0.05)
        st.success("Relaxation complete!")

        st.write("**Relaxed Structure:**")
        view(atoms)

# ================================
# 6. FOOTER
# ================================
st.markdown("Created by Sebin Devasia")
st.markdown("👨‍🔬 Built with ❤️ using **Streamlit**, **ASE**, **FairChem**, and **ORB**")
