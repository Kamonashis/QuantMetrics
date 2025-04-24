import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_correlation():
    st.title("📈 Correlation Matrix")

    tickers = st.text_area("Enter comma-separated ticker symbols", value="^BSESN, ^NSEI, ^NSEBANK, INR=X, GC=F, BZ=F, ^IXIC, ^DJI, ^GSPC")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    start = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end = st.date_input("End Date", pd.to_datetime("today"))

    if st.button("Calculate Correlation"):
        try:
            data = yf.download(tickers, start=start, end=end)['Close']
            corr_matrix = data.corr()

            st.dataframe(corr_matrix)

            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, annot_kws={"size": 7})
            ax.set_title("Correlation Matrix of Returns", fontsize=16)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
            plt.figure(figsize=(12, 10))
            st.pyplot(fig)


        except Exception as e:
            st.error(f"Failed to calculate correlation: {e}")