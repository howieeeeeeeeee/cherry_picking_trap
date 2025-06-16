from import_setup import *
from config.db import MONGOCLIENT_LOCAL, MONGOCLIENT
import streamlit as st


@st.cache_resource
def connect_to_db(db="cherry_picking_trap"):
    if os.environ.get("ENV") == "streamlit":
        return MONGOCLIENT[db]
    else:
        return MONGOCLIENT_LOCAL[db]


db = connect_to_db()
