import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau # Import for p-value calculation and other methods
from datetime import datetime, timedelta
import warnings

# Function to calculate the p-value matrix for Pearson correlation
def calculate_pvalue_matrix(df):
    """
    Calculates the pairwise p-values for Pearson correlation coefficients in a DataFrame.
    """
    df = df.dropna()._get_numeric_data() # Drop NaNs and get only numeric columns
    cols = df.columns
    pvalues_matrix = pd.DataFrame(index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            col1 = df[cols[i]]
            col2 = df[cols[j]]
            # Ensure there are enough non-NaN observations for the pair (at least 2 for correlation)
            if len(col1.dropna()) > 1 and len(col2.dropna()) > 1:
                 # Calculate Pearsonr which returns correlation and p-value
                corr, pvalue = pearsonr(col1, col2)
                pvalues_matrix.loc[cols[i], cols[j]] = pvalue
                pvalues_matrix.loc[cols[j], cols[i]] = pvalue # Matrix is symmetric
            else:
                pvalues_matrix.loc[cols[i], cols[j]] = np.nan
                pvalues_matrix.loc[cols[j], cols[i]] = np.nan
    return pvalues_matrix


def show_correlation():
    st.title("📈 Static Correlation Analysis")
    st.markdown("Analyze the static correlation between different financial assets.")

    # --- Input Section ---
    st.header("Input Parameters")
    # Use session state to persist ticker input value
    if 'corr_tickers_input' not in st.session_state:
        st.session_state['corr_tickers_input'] = "^BSESN, ^NSEI, ^NSEBANK, INR=X, GC=F, BZ=F, ^IXIC, ^DJI, ^GSPC"
    tickers_input = st.text_area("Enter comma-separated ticker symbols", value=st.session_state['corr_tickers_input'], key='corr_tickers_input_widget')
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    st.session_state['corr_tickers_input'] = tickers_input # Update session state

    col1, col2 = st.columns(2)
    with col1:
        # Use session state to persist start date
        if 'corr_start_date' not in st.session_state:
            st.session_state['corr_start_date'] = pd.to_datetime("2022-01-01").date() # Store as date
        start_date = st.date_input("Start Date", value=st.session_state['corr_start_date'], key='corr_start_date_widget')
        st.session_state['corr_start_date'] = start_date # Update session state

    with col2:
        # Use session state to persist end date
        if 'corr_end_date' not in st.session_state:
            st.session_state['corr_end_date'] = pd.to_datetime("today").date() # Store as date
        end_date = st.date_input("End Date", value=st.session_state['corr_end_date'], key='corr_end_date_widget')
        st.session_state['corr_end_date'] = end_date # Update session state

    st.subheader("Analysis Options")
    col3, col4 = st.columns(2)
    with col3:
        # Use session state for data type radio button
        if 'corr_data_type' not in st.session_state:
            st.session_state['corr_data_type'] = 'Close Price'
        data_type = st.radio("Analyze based on:", ('Close Price', 'Returns'), key='corr_data_type_widget')
        st.session_state['corr_data_type'] = data_type # Update session state

    with col4:
        # Use session state for correlation method selectbox
        if 'corr_method' not in st.session_state:
            st.session_state['corr_method'] = 'pearson'
        correlation_method = st.selectbox("Correlation Method:", ('pearson', 'spearman', 'kendall'), key='corr_method_widget')
        st.session_state['corr_method'] = correlation_method # Update session state

    # --- Run Analysis Button ---
    if st.button("Run Analysis", key='run_corr_analysis_button'):
        if len(tickers) < 2:
            st.error("Please enter at least 2 ticker symbols to calculate correlation.")
            # Clear previous results if button is clicked with insufficient tickers
            if 'corr_analysis_results' in st.session_state:
                del st.session_state['corr_analysis_results']
            return

        try:
            # Download 'Close' prices
            # Convert date objects back to datetime for yfinance
            data = yf.download(tickers, start=datetime.combine(start_date, datetime.min.time()), end=datetime.combine(end_date, datetime.min.time()))['Close']

            if data.empty:
                st.error("No data retrieved for the selected tickers and date range.")
                if 'corr_analysis_results' in st.session_state:
                    del st.session_state['corr_analysis_results']
                return

            # Prepare data based on user selection
            if data_type == 'Returns':
                processed_data = data.pct_change().dropna()
                data_label = "Returns"
            else: # Close Price
                processed_data = data.dropna()
                data_label = "Close Price"

            if processed_data.empty:
                 st.error(f"No valid {data_label} data available for the selected tickers and date range after processing.")
                 if 'corr_analysis_results' in st.session_state:
                    del st.session_state['corr_analysis_results']
                 return

            # --- Perform Static Correlation Analysis and Store Results ---
            st.subheader(f"📊 Static Correlation Matrix ({data_label} - {correlation_method.capitalize()})")

            # Calculate correlation matrix
            corr_matrix = processed_data.corr(method=correlation_method)

            # Store results in session state
            st.session_state['corr_analysis_results'] = {
                'corr_matrix': corr_matrix,
                'data_label': data_label,
                'correlation_method': correlation_method,
                'processed_data': processed_data, # Store processed data for scatter plot
                'tickers': tickers # Store tickers for scatter plot selectboxes
            }

            # Display correlation matrix
            st.dataframe(corr_matrix.style.format("{:.4f}"))

            # Calculate and display p-value matrix for Pearson correlation
            if correlation_method == 'pearson':
                st.subheader("🔬 P-value Matrix (Pearson Correlation)")
                st.markdown("P-values indicate the statistical significance of the correlation coefficient. A low p-value (e.g., < 0.05) suggests a statistically significant correlation.")
                pvalue_matrix = calculate_pvalue_matrix(processed_data)
                st.dataframe(pvalue_matrix.style.format("{:.4f}"))


            # Display heatmap
            st.subheader("Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, annot_kws={"size": 7})
            ax.set_title(f"Correlation Matrix of {data_label} ({correlation_method.capitalize()})", fontsize=14)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
            plt.tight_layout() # Adjust layout to prevent labels overlapping
            st.pyplot(fig)

            st.success("Analysis complete!")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            # Clear results on error
            if 'corr_analysis_results' in st.session_state:
                del st.session_state['corr_analysis_results']


    # --- Display Results (if available in session state) ---
    # Check if analysis_results exists and is not empty
    if 'corr_analysis_results' in st.session_state and st.session_state['corr_analysis_results']:
        results = st.session_state['corr_analysis_results']

        # Robustly retrieve data from the results dictionary
        corr_matrix = results.get('corr_matrix')
        data_label = results.get('data_label')
        correlation_method = results.get('correlation_method')
        processed_data = results.get('processed_data')
        tickers_for_scatter = results.get('tickers') # Use stored tickers for scatter plot

        # Check if essential data is present for display
        if corr_matrix is None or data_label is None or correlation_method is None or processed_data is None or tickers_for_scatter is None:
            st.warning("Incomplete analysis results found in session state. Please run the analysis again.")
            return

        st.subheader(f"📊 Static Correlation Matrix ({data_label} - {correlation_method.capitalize()})")
        st.dataframe(corr_matrix.style.format("{:.4f}"))

        # Display p-value matrix if it exists (only for Pearson)
        if correlation_method == 'pearson' and 'pvalue_matrix' in results:
             st.subheader("🔬 P-value Matrix (Pearson Correlation)")
             st.markdown("P-values indicate the statistical significance of the correlation coefficient. A low p-value (e.g., < 0.05) suggests a statistically significant correlation.")
             st.dataframe(results['pvalue_matrix'].style.format("{:.4f}"))


        st.subheader("Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, annot_kws={"size": 7})
        ax.set_title(f"Correlation Matrix of {data_label} ({correlation_method.capitalize()})", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        plt.tight_layout()
        st.pyplot(fig)


        # --- Scatter Plot Section ---
        st.markdown("---")
        st.subheader("Scatter Plot of Ticker Pair")
        st.markdown(f"Visualize the relationship between two tickers based on {data_label}.")

        # Use stored tickers and unique keys for selectboxes
        scatter_ticker1 = st.selectbox(f"Select Ticker for X-axis ({data_label}):", tickers_for_scatter, key='scatter_ticker1_display_corr')
        scatter_ticker2 = st.selectbox(f"Select Ticker for Y-axis ({data_label}):", tickers_for_scatter, key='scatter_ticker2_display_corr')

        if scatter_ticker1 and scatter_ticker2 and scatter_ticker1 != scatter_ticker2:
            # Ensure selected tickers are in processed_data columns
            if scatter_ticker1 in processed_data.columns and scatter_ticker2 in processed_data.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                # Use the processed_data retrieved from session state
                ax.scatter(processed_data[scatter_ticker1], processed_data[scatter_ticker2], alpha=0.5)
                ax.set_title(f"Scatter Plot: {scatter_ticker1} vs {scatter_ticker2} ({data_label})")
                ax.set_xlabel(f"{scatter_ticker1} {data_label}")
                ax.set_ylabel(f"{scatter_ticker2} {data_label}")
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.warning("Selected tickers not found in the processed data.")
        elif scatter_ticker1 == scatter_ticker2:
            st.info("Please select two different tickers for the scatter plot.")
        else:
            st.info("Select a ticker pair to see the scatter plot.")