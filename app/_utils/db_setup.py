from import_setup import *
import streamlit as st


@st.cache_resource
def connect_to_db(db="cherry_picking_trap"):
    if os.environ.get("ENV") == "streamlit":
        from config.db import MONGOCLIENT

        return MONGOCLIENT[db]
    else:
        from config.db import MONGOCLIENT_LOCAL

        return MONGOCLIENT_LOCAL[db]


db = connect_to_db()
