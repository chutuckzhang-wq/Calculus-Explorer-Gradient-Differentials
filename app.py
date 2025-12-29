import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Calculus Assignment", layout="wide")

# Custom CSS for the "Nicer" Dashboard look
st.markdown("""
<style>
    .metric-label {
        font-size: 14px;
        font-weight: 600;
        color: #b0b0b0;
        margin-bottom: 2px;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    .stCode {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("1. Position Controls (Topic 4)")
example_choice = st.sidebar.radio(
    "Select Surface:",
    ["Simple: Inverted Paraboloid", "Complex: Multi-Peak Sine Wave"]
)

st.sidebar.write("**Current Position (x, y):**")
x_val = st.sidebar.slider("X Coordinate", -2.0, 2.0, 0.5, 0.1)
y_val = st.sidebar.slider("Y Coordinate", -2.0, 2.0, 0.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("2. Differentials Controls (Topic 5)")
st.sidebar.write("**Step size for estimation:**")
dx_val = st.sidebar.slider("Change in x (dx)", -0.5, 0.5, 0.1, 0.01)
dy_val = st.sidebar.slider("Change in y (dy)", -0.5, 0.5, 0.1, 0.01)

# --- NEW: TEAM NAMES ---
st.sidebar.markdown("---")
st.sidebar.subheader("Project Team")
st.sidebar.write("1. **CHU TUCK ZHANG**")
st.sidebar.write("2. **JOEY LEE XIN PEI**")
st.sidebar.write("3. **LEI AI QIN**")

# --- 3. MATHEMATICAL CALCULATIONS ---
# Generate grid for 3D Surface
x_range = np.linspace(-2.5, 2.5, 50)
y_range = np.linspace(-2.5, 2.5, 50)
X, Y = np.meshgrid(x_range, y_range)

if example_choice == "Simple: Inverted Paraboloid":
    # z = 4 - x^2 - y^2
    Z = 4 - X ** 2 - Y ** 2

    def get_z(x, y): return 4 - x ** 2 - y ** 2

    # Partial Derivatives
    # dz/dx = -2x, dz/dy = -2y
    dx = -2 * x_val
    dy = -2 * y_val

else:
    # z = sin(2x) + cos(2y)
    Z = np.sin(X * 2) + np.cos(Y * 2)

    def get_z(x, y): return np.sin(x * 2) + np.cos(y * 2)

    # Partial Derivatives
    dx = 2 * np.cos(x_val * 2)
    dy = -2 * np.sin(y_val * 2)

# Calculate Points
current_z = get_z(x_val, y_val) # Red Dot

# Topic 5 Calculations
# A. Linear Approximation (dz)
total_differential_dz = (dx * dx_val) + (dy * dy_val)

# B. Actual Result (New Point / Green Dot)
new_z_actual = get_z(x_val + dx_val, y_val + dy_val)
actual_change_delta_z = new_z_actual - current_z

# --- 4. VISUALIZATION (PLOTLY) ---
fig = go.Figure()

# Surface
fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name="Terrain"))

# Start Point (Red)
fig.add_trace(go.Scatter3d(
    x=[x_val], y=[y_val], z=[current_z],
    mode='markers', marker=dict(size=8, color='red'), name='Start Point'
))

# New Point (Green) - "The Feature"
fig.add_trace(go.Scatter3d(
    x=[x_val + dx_val], y=[y_val + dy_val], z=[new_z_actual],
    mode='markers', marker=dict(size=8, color='green'), name='New Point'
))

# Gradient Vector (Orange Arrow)
scale = 0.5
fig.add_trace(go.Scatter3d(
    x=[x_val, x_val + dx * scale],
    y=[y_val, y_val + dy * scale],
    z=[current_z, current_z + np.sqrt(dx ** 2 + dy ** 2) * scale],
    mode='lines+markers', line=dict(color='orange', width=5),
    marker=dict(size=4, color='orange'), name='Gradient Vector'
))

fig.update_layout(title=f"3D Visualization: {example_choice}", height=600, margin=dict(l=0, r=0, b=0, t=40))

# --- 5. MAIN DASHBOARD ---
st.title("Calculus Explorer: Gradient & Differentials")

col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # === SECTION 1: GRADIENT ===
    st.markdown('<div class="section-header">1. Gradient (Steepest Ascent)</div>', unsafe_allow_html=True)
    st.latex(r"\nabla f = \langle f_x, f_y \rangle")
    
    # Nested columns for Partials
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown('<div class="metric-label">Partial X (Slope in x)</div>', unsafe_allow_html=True)
        st.code(f"{dx:.4f}")
    with g_col2:
        st.markdown('<div class="metric-label">Partial Y (Slope in y)</div>', unsafe_allow_html=True)
        st.code(f"{dy:.4f}")
    
    # Steepness
    st.markdown('<div class="metric-label">Steepness (Magnitude)</div>', unsafe_allow_html=True)
    st.code(f"{np.sqrt(dx**2 + dy**2):.4f}")
    
    st.info("💡 **Visual Guide:** The **Orange Arrow** uses these values to point uphill (Steepest Ascent).")

    # === SECTION 2: DIFFERENTIALS ===
    st.markdown('<div class="section-header">2. Differentials (Approximation)</div>', unsafe_allow_html=True)
    
    st.latex(r"dz = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy")

    # Nested columns for Comparison
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown('<div class="metric-label">Estimate (dz)</div>', unsafe_allow_html=True)
        st.code(f"{total_differential_dz:.4f}")
    with d_col2:
        st.markdown('<div class="metric-label">Actual (Δz)</div>', unsafe_allow_html=True)
        st.code(f"{actual_change_delta_z:.4f}")

    # The Green Dot Tip (Requested Feature)
    st.info("""
    **Visual Guide:** The **Green Dot** shows where you land on the *actual* curved surface.
    Compare its height to the **Red Dot** to see the Actual Change ($\Delta z$).
    """)

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

# --- 6. REAL WORLD APPLICATION & FEATURE EXPLANATION ---
st.divider()
st.subheader("Real World Significance & Feature Analysis")

st.markdown("""
### 1. The "New Point" Feature: Engineering Tolerance
To facilitate error analysis, this application visualizes the post-displacement coordinate $f(x+dx, y+dy)$ as a distinct **Green Marker**. 
* **Formal Explanation:** This feature enables a direct comparative analysis between the linear approximation provided by the total differential ($dz$) and the true non-linear behavior of the surface ($\Delta z$).
* **Real World Use:** In manufacturing, this concept is used for **Tolerance Analysis**. If a machine cuts a part with a tiny error ($dx$), engineers use the Differential ($dz$) to predict how much that tiny error will propagate to the final product's volume or strength, ensuring safety margins are met.

### 2. The Gradient Feature: Optimization & AI
The orange gradient vector always points in the direction of steepest ascent.
* **Real World Use:** This is the visual representation of **Gradient Descent**, the algorithm used to train Artificial Intelligence models. AI models calculate this vector to find the "path of steepest descent" to minimize their error rates and learn from data.

### 3. Sensitivity Analysis (Finance)
Financial analysts use partial derivatives (displayed in the dashboard) to determine which variables (e.g., Interest Rates vs. Oil Prices) have the largest impact on a portfolio's value, allowing for better risk management.
""")
Financial analysts use partial derivatives (displayed in the dashboard) to determine which variables (e.g., Interest Rates vs. Oil Prices) have the largest impact on a portfolio's value, allowing for better risk management.

""")
