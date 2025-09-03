import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

# Define case markers and colors
CASE_MARKERS = {
    "Case A: Always Bad": "x",
    "Case B: Always Good": "triangle-up",
    "Case C: Interesting Case": "diamond",
    "Case D: Always Informed": "square",
}

CASE_COLORS = {
    "Case A: Always Bad": "#bdbdbd",  # Medium grey
    "Case B: Always Good": "#bdbdbd",  # Light grey
    "Case C: Interesting Case": "#bdbdbd",  # Very light grey
    "Case D: Always Informed": "#bdbdbd",  # Dark grey
}


def plot_simulation_results(
    df: pd.DataFrame, lamb: float = None, scale: float = None
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    Create plots for simulation results including welfare and equilibrium values.

    Args:
        df: DataFrame containing simulation results
        lamb: Lambda value for the title
        scale: Scale value for the title

    Returns:
        tuple[go.Figure, go.Figure, go.Figure, go.Figure]: Tuple of figures for welfare, alpha_star, q_star, and moderate policies
    """

    marker_size = 8 if len(df["K"]) < 60 else 4
    # Create welfare line plot
    welfare_fig = go.Figure()

    # Add proposer payoff line
    welfare_fig.add_trace(
        go.Scatter(
            x=df["K"],
            y=df["proposer_expected_payoff"],
            mode="lines+markers",
            name="Proposer Payoff",
            line=dict(color="#7FB3D5", width=2),  # Soft blue
            marker=dict(
                size=marker_size,
                symbol=[CASE_MARKERS[case] for case in df["case"]],
                color=[CASE_COLORS[case] for case in df["case"]],
            ),
        )
    )

    # Add responder payoff line
    welfare_fig.add_trace(
        go.Scatter(
            x=df["K"],
            y=df["responder_expected_payoff"],
            mode="lines+markers",
            name="Responder Payoff",
            line=dict(color="#F5B7B1", width=2),  # Soft pink
            marker=dict(
                size=marker_size,
                symbol=[CASE_MARKERS[case] for case in df["case"]],
                color=[CASE_COLORS[case] for case in df["case"]],
            ),
        )
    )

    # Add total welfare line
    total_welfare = df["proposer_expected_payoff"] + df["responder_expected_payoff"]
    welfare_fig.add_trace(
        go.Scatter(
            x=df["K"],
            y=total_welfare,
            mode="lines+markers",
            name="Total Welfare",
            line=dict(color="#2E86C1", width=2, dash="dash"),  # Dashed blue
            marker=dict(
                size=marker_size,
                symbol=[CASE_MARKERS[case] for case in df["case"]],
                color=[CASE_COLORS[case] for case in df["case"]],
            ),
            visible="legendonly",
        )
    )

    # Add case markers legend
    for case, marker in CASE_MARKERS.items():
        if case in df["case"].unique():
            welfare_fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=case,
                    marker=dict(symbol=marker, size=10, color=CASE_COLORS[case]),
                    showlegend=True,
                )
            )

    # Create title with lambda and scale if provided
    title = "Ex-Ante Welfare by K"
    if lamb is not None and scale is not None:
        title = f"Ex-Ante Welfare by K (λ={lamb}, μ={scale})"

    welfare_fig.update_layout(
        title=title,
        xaxis_title="K",
        yaxis_title="Expected Payoff",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,  # Increased height to accommodate legend
    )

    # Create alpha_star line plot
    alpha_fig = go.Figure()
    alpha_fig.add_trace(
        go.Scatter(
            x=df["K"],
            y=df["alpha_star"],
            mode="lines+markers",
            name="α*",
            line=dict(color="#636363", width=2),  # Modern grey
            marker=dict(
                size=8,
                symbol=[CASE_MARKERS[case] for case in df["case"]],
                color=[CASE_COLORS[case] for case in df["case"]],
            ),
        )
    )
    alpha_fig.update_layout(
        title="Optimal α* by K",
        xaxis_title="K",
        yaxis_title="α*",
        template="plotly_white",
        height=150,
        margin=dict(t=30, b=10),  # Reduce vertical margins
    )

    # Create q_star line plot
    q_fig = go.Figure()
    q_fig.add_trace(
        go.Scatter(
            x=df["K"],
            y=df["q_star"],
            mode="lines+markers",
            name="q*",
            line=dict(color="#636363", width=2),  # Modern grey
            marker=dict(
                size=8,
                symbol=[CASE_MARKERS[case] for case in df["case"]],
                color=[CASE_COLORS[case] for case in df["case"]],
            ),
        )
    )
    q_fig.update_layout(
        title="Optimal q* by K",
        xaxis_title="K",
        yaxis_title="q*",
        template="plotly_white",
        height=150,
        margin=dict(t=30, b=10),  # Reduce vertical margins
    )

    # Create moderate policies percentage plot
    moderate_fig = go.Figure()
    moderate_fig.add_trace(
        go.Scatter(
            x=df["K"],
            y=df["percent_of_moderate_policies"] * 100,  # Convert to percentage
            mode="lines+markers",
            name="Moderate Policies %",
            line=dict(color="#636363", width=2),  # Modern grey
            marker=dict(
                size=8,
                symbol=[CASE_MARKERS[case] for case in df["case"]],
                color=[CASE_COLORS[case] for case in df["case"]],
            ),
        )
    )
    moderate_fig.update_layout(
        title="Percentage of Moderate Policies (q < q*) by K",
        xaxis_title="K",
        yaxis_title="Percentage (%)",
        template="plotly_white",
        height=150,
        margin=dict(t=30, b=10),  # Reduce vertical margins
    )

    return welfare_fig, alpha_fig, q_fig, moderate_fig
