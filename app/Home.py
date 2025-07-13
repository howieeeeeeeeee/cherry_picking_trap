import streamlit as st
from import_setup import *


st.set_page_config(layout="wide", initial_sidebar_state="auto")


summary = st.Page(
    "reports/summary.py", title="Summary", icon=":material/dashboard:", default=True
)
methodology = st.Page(
    "reports/methodology.py", title="Methodology", icon=":material/dashboard:"
)

anecdote = st.Page("reports/anecdote.py", title="Anecdote", icon=":material/history:")

simulate = st.Page("tools/simulate.py", title="Simulate", icon=":material/calculate:")


pg = st.navigation(
    {
        "Reports": [summary, methodology, anecdote],
        # "Tools": [simulate],
    }
)

pg.run()
