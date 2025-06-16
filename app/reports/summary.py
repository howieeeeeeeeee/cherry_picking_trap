import streamlit as st
from app._utils.db_setup import db
from src.data_management.mongo_handler import get_simulation_results_filters
from src.analysis.tables import get_simulation_data
from src.analysis.graphs import plot_simulation_results

st.markdown("### Cherry Picking Trap - Simulation Results Analysis")

# Get available filters
filters = get_simulation_results_filters(db)

# Create selection widgets
col1, col2 = st.columns(2)
with col1:
    selected_lamb = st.selectbox("Select Lambda", sorted(filters["lamb"]))
with col2:
    selected_scale = st.selectbox("Select Scale", sorted(filters["scale"]))

with st.expander("Other Settings"):
    show_full_K_range = st.toggle("Show full K range", False)
    if show_full_K_range:
        K_range = [1, max(filters["K"])]
    else:
        K_range = [1, 20]


# Add a button to generate the plot
if st.button("Generate Plots", use_container_width=True):
    # Get data
    df = get_simulation_data(db, selected_lamb, selected_scale, K_range)

    if not df.empty:
        # Create and display the plots
        welfare_fig, alpha_fig, q_fig = plot_simulation_results(df)

        # Display welfare plot
        st.plotly_chart(welfare_fig, use_container_width=True)

        # Display alpha and q plots vertically
        st.plotly_chart(alpha_fig, use_container_width=True)
        st.plotly_chart(q_fig, use_container_width=True)

        # Display raw data in an expander
        with st.expander("View Raw Data"):
            # Round the values to 4 decimal places
            display_df = df.round(4)
            st.dataframe(display_df)
    else:
        st.warning("No data available for the selected parameters.")

## Welfare and Equilibirum under different K
filters = get_simulation_results_filters(db)

## allow user to select lamb, scale, one for each
