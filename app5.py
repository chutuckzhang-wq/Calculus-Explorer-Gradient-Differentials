import streamlit as st
import numpy as np
import plotly.graph_objects as go
import re

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Calculus Assignment", layout="wide")

st.markdown("""
<style>
    .metric-label { font-size: 14px; font-weight: 600; color: #b0b0b0; margin-bottom: 2px; }
    .section-header { font-size: 18px; font-weight: bold; margin-top: 20px; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    .stCode { font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)


# --- HELPER: CONVERT DESMOS STYLE TO PYTHON ---
def preprocess_formula(formula):
    """
    Converts user-friendly math (Desmos style) into Python numpy syntax.
    Example: "x^2 + sin(y)" -> "x**2 + np.sin(y)"
    """
    # 1. Replace carat '^' with python power '**'
    formula = formula.replace("^", "**")

    # 2. Add 'np.' to common math functions if missing
    math_functions = [
        "sin", "cos", "tan", "sqrt", "exp", "log", "abs",
        "pi", "e", "arcsin", "arccos", "arctan"
    ]

    for func in math_functions:
        pattern = r'(?<!\.)\b' + func + r'\b'
        formula = re.sub(pattern, f"np.{func}", formula)

    return formula


# --- 2. SIDEBAR: SETUP & AXIS CONTROL ---
st.sidebar.header("1. Axis & Grid Setup")
st.sidebar.caption("Set the Min/Max range for the visualization.")

col_min, col_max = st.sidebar.columns(2)
with col_min:
    axis_min = st.number_input("Min Axis", value=-2.5, step=0.5)
with col_max:
    axis_max = st.number_input("Max Axis", value=2.5, step=0.5)

if axis_min >= axis_max:
    st.sidebar.error("Max must be greater than Min!")
    axis_max = axis_min + 1.0

st.sidebar.markdown("---")

# --- 3. SIDEBAR: FUNCTION SELECTION ---
st.sidebar.header("2. Surface Function")
example_choice = st.sidebar.radio(
    "Select or Type Function:",
    ["Simple: Inverted Paraboloid", "Complex: Multi-Peak Sine Wave", "Custom: User Defined Input"]
)

custom_formula_display = ""
final_formula_code = ""

if example_choice == "Custom: User Defined Input":
    st.sidebar.info("💡 Type normally! (e.g., x^2 + y^2, sin(x)*cos(y))")
    user_input = st.sidebar.text_input("Enter z = f(x, y)", value="sin(x) * cos(y)")

    try:
        final_formula_code = preprocess_formula(user_input)
        custom_formula_display = user_input
    except:
        final_formula_code = "0"

st.sidebar.markdown("---")

# --- 4. SIDEBAR: POSITION CONTROLS ---
st.sidebar.header("3. Position Controls")

# UPDATED: Initial Position set to 0.0
x_val = st.sidebar.slider("X Coordinate", float(axis_min), float(axis_max), 0.0, 0.1)
y_val = st.sidebar.slider("Y Coordinate", float(axis_min), float(axis_max), 0.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("4. Differentials Controls")

# UPDATED: Initial Change (dx, dy) set to 0.0
dx_val = st.sidebar.slider("Change in x (dx)", -0.5, 0.5, 0.0, 0.01)
dy_val = st.sidebar.slider("Change in y (dy)", -0.5, 0.5, 0.0, 0.01)

# --- TEAM NAMES ---
st.sidebar.markdown("---")
st.sidebar.subheader("Project Team")
st.sidebar.write("1. **CHU TUCK ZHANG**")
st.sidebar.write("2. **JOEY LEE XIN PEI**")
st.sidebar.write("3. **LEI AI QIN**")

# --- 5. MATHEMATICAL LOGIC ---

# A. Generate Grid
x_range = np.linspace(axis_min, axis_max, 60)
y_range = np.linspace(axis_min, axis_max, 60)
X, Y = np.meshgrid(x_range, y_range)

# B. Define Function
error_msg = None

try:
    if example_choice == "Simple: Inverted Paraboloid":
        Z = 4 - X ** 2 - Y ** 2


        def get_z_val(x, y):
            return 4 - x ** 2 - y ** 2

    elif example_choice == "Complex: Multi-Peak Sine Wave":
        Z = np.sin(X * 2) + np.cos(Y * 2)


        def get_z_val(x, y):
            return np.sin(x * 2) + np.cos(y * 2)

    else:  # CUSTOM INPUT
        safe_dict = {"x": X, "y": Y, "np": np}
        Z = eval(final_formula_code, {"__builtins__": None}, safe_dict)


        def get_z_val(x, y):
            return eval(final_formula_code, {"__builtins__": None}, {"x": x, "y": y, "np": np})

except Exception as e:
    error_msg = f"Invalid Formula. Try 'x^2 + y^2'. Error: {e}"
    Z = np.zeros_like(X)


    def get_z_val(x, y):
        return 0.0

# C. Calculate Derivatives (Numerical)
h = 0.0001
current_z = get_z_val(x_val, y_val)

# Partial X
z_plus_x = get_z_val(x_val + h, y_val)
z_minus_x = get_z_val(x_val - h, y_val)
dx_slope = (z_plus_x - z_minus_x) / (2 * h)

# Partial Y
z_plus_y = get_z_val(x_val, y_val + h)
z_minus_y = get_z_val(x_val, y_val - h)
dy_slope = (z_plus_y - z_minus_y) / (2 * h)

# D. Differentials
total_differential_dz = (dx_slope * dx_val) + (dy_slope * dy_val)
new_z_actual = get_z_val(x_val + dx_val, y_val + dy_val)
actual_change_delta_z = new_z_actual - current_z

# --- 6. VISUALIZATION ---
st.title("Calculus Explorer: Gradient & Differentials")

if error_msg:
    st.error(error_msg)
    st.stop()

col1, col2 = st.columns([3, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name="Terrain"))

    # Start Point
    fig.add_trace(go.Scatter3d(
        x=[x_val], y=[y_val], z=[current_z],
        mode='markers', marker=dict(size=8, color='red'), name='Start Point'
    ))

    # New Point
    fig.add_trace(go.Scatter3d(
        x=[x_val + dx_val], y=[y_val + dy_val], z=[new_z_actual],
        mode='markers', marker=dict(size=8, color='green'), name='New Point'
    ))

    # Gradient Vector
    scale = 0.5
    fig.add_trace(go.Scatter3d(
        x=[x_val, x_val + dx_slope * scale],
        y=[y_val, y_val + dy_slope * scale],
        z=[current_z, current_z + np.sqrt(dx_slope ** 2 + dy_slope ** 2) * scale],
        mode='lines+markers', line=dict(color='orange', width=5),
        marker=dict(size=4, color='orange'), name='Gradient Vector'
    ))

    fig.update_layout(title="3D Interactive Surface", height=600, margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # === SECTION 1: GRADIENT ===
    st.markdown('<div class="section-header">1. Gradient (Steepest Ascent)</div>', unsafe_allow_html=True)

    st.latex(r"\nabla f = \langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \rangle")

    # Nested columns for Partials
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown('<div class="metric-label">Partial X (Slope in x)</div>', unsafe_allow_html=True)
        st.code(f"{dx_slope:.4f}")
    with g_col2:
        st.markdown('<div class="metric-label">Partial Y (Slope in y)</div>', unsafe_allow_html=True)
        st.code(f"{dy_slope:.4f}")

    # Steepness
    st.markdown('<div class="metric-label">Steepness (Magnitude)</div>', unsafe_allow_html=True)
    st.code(f"{np.sqrt(dx_slope ** 2 + dy_slope ** 2):.4f}")

    st.info("💡 **Visual Guide:** The **Orange Arrow** uses these values to point uphill (Steepest Ascent).")

    # === SECTION 2: DIFFERENTIALS ===
    st.markdown('<div class="section-header">2. Differentials (Approximation)</div>', unsafe_allow_html=True)

    st.latex(r"dz = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy")

    # Nested columns for Comparison
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="metric-label">Estimate (dz)</div>', unsafe_allow_html=True)
        st.code(f"{total_differential_dz:.4f}")
    with d2:
        st.markdown('<div class="metric-label">Actual (Δz)</div>', unsafe_allow_html=True)
        st.code(f"{actual_change_delta_z:.4f}")

    # Accuracy Visualization
    difference = abs(total_differential_dz - actual_change_delta_z)
    st.markdown('<div class="metric-label">Approximation Error</div>', unsafe_allow_html=True)

    # Health Bar Logic
    accuracy_score = max(0, 1.0 - difference)
    if difference < 0.1:
        msg = "Excellent Approximation"
        st.progress(accuracy_score)
        st.caption(f"**{msg}** (Error: {difference:.4f})")
    else:
        st.progress(accuracy_score)
        st.warning(f"High Error: {difference:.4f}")
        st.caption("Step size is too large for linear approximation.")

    # The Green Dot Tip
    st.info("""
    💡 **Visual Guide:** The **Green Dot** shows where you land on the *actual* curved surface.
    Compare its height to the **Red Dot** to see the Actual Change ($\Delta z$).
    """)

# --- 7. FOOTER EXPLANATION ---
st.divider()
st.subheader("Real World Significance & Feature Analysis")

st.markdown("""
### 1. The "Green Dot" Feature: Tolerance & Error Analysis (Manufacturing)
In the real world, "perfect" measurements are impossible.
* **The Concept:** This app visualizes the difference between a linear prediction ($dz$) and the actual result ($\Delta z$).
* **Real World Application:** Manufacturing engineers use this to set **Tolerances**. If a machine part varies slightly in size ($dx$), they use differentials to predict if the final product will still fit and function correctly. The "Green Dot" visualizes this reality check, ensuring safety margins are met without needing complex re-calculations.

### 2. The Gradient Feature: Optimization (Artificial Intelligence)
The orange arrow in this app always points in the direction of steepest ascent.
* **The Concept:** This vector shows the path of fastest change.
* **Real World Application:** This is the visual basis of **Gradient Descent**, the algorithm used to train AI models. Just as the arrow points uphill, AI algorithms calculate this vector to find the "downhill" path to minimize errors and learn from data efficiently.

### 3. Sensitivity Analysis (Chemical Engineering)
The partial derivatives displayed in the dashboard measure how sensitive a function is to changes in its inputs.
* **The Concept:** $\\frac{\\partial f}{\\partial x}$ (Partial X) tells us how much the output changes if we tweak just one input.
* **Real World Application:** In chemical plants, the **Yield** of a reaction depends on variables like **Temperature** ($x$) and **Pressure** ($y$). The partial derivatives tell engineers which variable has the biggest impact. If the derivative for Temperature is high, they know that even a tiny change in heat will drastically alter the result, requiring strict safety controls.
""")