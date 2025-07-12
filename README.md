# The Cherry-Picking Trap: A Simulation and Analysis Framework

This repository contains a sophisticated simulation and analysis tool for the "Cherry-Picking Trap" game theory model. It provides a web-based interface to run Monte Carlo simulations, analyze the results, and visualize the equilibrium dynamics of the model.

## About the Model

The Cherry-Picking Trap is a game-theoretic model that explores the strategic interactions between a "Proposer" and a "Responder." The Proposer has the option to propose either a "moderate" or a "cherry-picked" policy from a given set of options. The Responder, who may be informed or uninformed about the nature of the proposed policy, can then choose to accept or reject it. This model is particularly relevant for understanding political and economic scenarios where information asymmetry plays a crucial role.

## Features

*   **High-Performance Simulation:** The core of the project is a high-performance Monte Carlo simulation engine built with Python, `numpy`, and `numba` for accelerated computations.
*   **Interactive Web Application:** A user-friendly web application built with `streamlit` allows for easy configuration and execution of simulations.
*   **Advanced Analytics:** The framework uses `scipy` for numerical optimization to solve for game-theoretic equilibria.
*   **Data Visualization:** The application includes interactive plots and tables for visualizing simulation results, powered by `plotly` and `seaborn`.
*   **Flexible Data Generation:** The model supports various statistical distributions (Exponential, Poisson, Uniform) for generating policy baskets.

## Data Workflow

The current data workflow is as follows:

1.  **Simulation:** Simulations are run locally using the IPython notebook (`notebooks/test.ipynb`).
2.  **Local Storage:** The simulation results are saved to a local MongoDB instance.
3.  **Data Transfer:** The data is then transferred from the local MongoDB to an online MongoDB instance.
4.  **Online Application:** The deployed Streamlit application reads the data from the online MongoDB instance to populate the reports and visualizations.

## Getting Started

### Prerequisites

*   Python 3.11
*   `pipenv` for managing dependencies.
*   MongoDB (local and online instances)

### Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cherry_picking_trap
    ```

2.  **Install dependencies using pipenv:**
    ```bash
    pipenv install
    pipenv shell
    ```

### Running the Application

To launch the Streamlit web application, run the following command from the project's root directory:

```bash
streamlit run app/Home.py
```

This will open the application in your default web browser.

## Project Structure

```
├── app/                # Contains the Streamlit application code
│   ├── Home.py         # Main entry point for the Streamlit app
│   ├── reports/        # Pages for displaying reports and methodology
│   └── tools/          # Pages for interactive simulation tools
├── src/                # Source code for the simulation and analysis
│   ├── simulation/     # Core simulation model and logic
│   ├── data_management/ # Scripts for managing data (e.g., MongoDB handlers)
│   └── analysis/       # Scripts for data analysis and visualization
├── data/               # Directory for storing simulation results
├── notebooks/          # Jupyter notebooks for exploratory analysis and simulation
├── tests/              # (WIP) Unit and integration tests
├── main.py             # Main entry point for the project (CLI)
├── Pipfile             # Project dependencies
└── README.md           # This file
```