from import_setup import *
from config.db import MONGOCLIENT_LOCAL
import streamlit as st


@st.cache_resource
def connect_to_db(db="cherry_picking_trap"):
    return MONGOCLIENT_LOCAL[db]


db = connect_to_db()
