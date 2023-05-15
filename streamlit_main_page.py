import streamlit as st
import f_trade_app_main # make sure this file is in the same directory or PYTHONPATH
import f_trade_app_4h # make sure this file is in the same directory or PYTHONPATH
import f_AI_model # make sure this file is in the same directory or PYTHONPATH

PAGES = {
    "Global View": f_trade_app_main,
    "Indicator Order": f_trade_app_4h,
    "Ask to AI": f_AI_model
}

def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    
    with st.spinner(f"Loading {selection} ..."):
        page.app()

if __name__ == "__main__":
    main()
