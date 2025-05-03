import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau # Import for p-value calculation and other methods

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

# Function to calculate rolling correlation for a pair of series
def calculate_rolling_correlation(series1, series2, window, method):
    """
    Calculates the rolling correlation between two pandas Series.
    """
    # Combine the two series into a DataFrame
    df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()

    if df.shape[0] < window:
        return pd.Series(dtype=float) # Return empty series if not enough data

    # Calculate rolling correlation
    # We apply a function to the rolling window that calculates the correlation
    rolling_corr = df.rolling(window).apply(
        lambda x: x['s1'].corr(x['s2'], method=method),
        raw=False # raw=False passes a DataFrame to the function
    )
    return rolling_corr.dropna()


def show_correlation():
    st.title("📈 Correlation Analysis")
    st.markdown("Analyze the correlation between different financial assets.")

    # --- Input Section ---
    st.header("Input Parameters")
    # Use session state to persist ticker input value
    if 'tickers_input' not in st.session_state:
        st.session_state['tickers_input'] = "^BSESN, ^NSEI, ^NSEBANK, INR=X, GC=F, BZ=F, ^IXIC, ^DJI, ^GSPC"
    tickers_input = st.text_area("Enter comma-separated ticker symbols", value=st.session_state['tickers_input'], key='tickers_input_widget')
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        # Use session state to persist start date
        if 'start_date' not in st.session_state:
            st.session_state['start_date'] = pd.to_datetime("2022-01-01")
        start_date = st.date_input("Start Date", value=st.session_state['start_date'], key='start_date_widget')
        st.session_state['start_date'] = start_date # Update session state

    with col2:
        # Use session state to persist end date
        if 'end_date' not in st.session_state:
            st.session_state['end_date'] = pd.to_datetime("today")
        end_date = st.date_input("End Date", value=st.session_state['end_date'], key='end_date_widget')
        st.session_state['end_date'] = end_date # Update session state


    st.subheader("Analysis Options")
    col3, col4 = st.columns(2)
    with col3:
        # Use session state for data type radio button
        if 'data_type' not in st.session_state:
            st.session_state['data_type'] = 'Close Price'
        data_type = st.radio("Analyze based on:", ('Close Price', 'Returns'), key='data_type_widget')
        st.session_state['data_type'] = data_type # Update session state

        # Use session state for correlation method selectbox
        if 'correlation_method' not in st.session_state:
            st.session_state['correlation_method'] = 'pearson'
        correlation_method = st.selectbox("Correlation Method:", ('pearson', 'spearman', 'kendall'), key='corr_method_widget')
        st.session_state['correlation_method'] = correlation_method # Update session state

    with col4:
        # Use session state for analysis type radio button
        if 'analysis_type' not in st.session_state:
            st.session_state['analysis_type'] = 'Static Correlation'
        analysis_type = st.radio("Select Analysis Type:", ('Static Correlation', 'Rolling Correlation', 'Volatility Correlation'), key='analysis_type_widget')
        st.session_state['analysis_type'] = analysis_type # Update session state


    window_size = None
    if analysis_type in ['Rolling Correlation', 'Volatility Correlation']:
        # Use session state for window size slider
        if 'window_size' not in st.session_state:
            st.session_state['window_size'] = 60
        window_size = st.slider("Rolling Window Size (days)", 30, 252, value=st.session_state['window_size'], key='window_size_widget')
        st.session_state['window_size'] = window_size # Update session state
        st.markdown(f"**Note:** Rolling calculations require a window size.")

    # --- Run Analysis Button ---
    # Add a key to the button to help Streamlit manage its state
    if st.button("Run Analysis", key='run_analysis_button'):
        if not tickers:
            st.error("Please enter ticker symbols.")
            # Clear previous results if button is clicked with no tickers
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            return

        try:
            # --- Data Loading and Preparation ---
            data = yf.download(tickers, start=start_date, end=end_date)['Close']

            if data.empty:
                st.error("No data retrieved for the selected tickers and date range.")
                if 'analysis_results' in st.session_state:
                    del st.session_state['analysis_results']
                return

            # Handle cases with only one ticker
            if len(tickers) == 1:
                 st.warning("Please enter more than one ticker to calculate correlation.")
                 st.dataframe(data) # Still show the price data for the single ticker
                 if 'analysis_results' in st.session_state:
                    del st.session_state['analysis_results']
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
                 if 'analysis_results' in st.session_state:
                    del st.session_state['analysis_results']
                 return

            # --- Perform Analysis based on Analysis Type and Store Results ---
            analysis_results = {}
            analysis_results['analysis_type'] = analysis_type
            analysis_results['data_label'] = data_label
            analysis_results['correlation_method'] = correlation_method
            analysis_results['processed_data'] = processed_data # Store processed data for scatter plot

            if analysis_type == 'Static Correlation':
                # Calculate correlation matrix
                corr_matrix = processed_data.corr(method=correlation_method)
                analysis_results['corr_matrix'] = corr_matrix

                # Calculate p-value matrix for Pearson correlation
                if correlation_method == 'pearson':
                    pvalue_matrix = calculate_pvalue_matrix(processed_data)
                    analysis_results['pvalue_matrix'] = pvalue_matrix

            elif analysis_type == 'Rolling Correlation':
                if processed_data.shape[0] < window_size:
                    st.warning(f"Not enough data points ({processed_data.shape[0]}) for the specified rolling window size ({window_size}). Please reduce the window size or extend the date range.")
                    if 'analysis_results' in st.session_state:
                        del st.session_state['analysis_results']
                    return
                analysis_results['window_size'] = window_size
                # Rolling correlation for all pairs is computationally expensive and the result is large.
                # We will calculate and store the processed data and window size,
                # and calculate the specific pair's rolling correlation only when selected for plotting.


            elif analysis_type == 'Volatility Correlation':
                analysis_results['window_size'] = window_size
                if data_type != 'Returns':
                    st.warning("Volatility correlation is typically calculated on returns. Switching to Returns for this analysis.")
                    returns_data = data.pct_change().dropna()
                    if returns_data.empty:
                        st.error("No return data available to calculate volatility.")
                        if 'analysis_results' in st.session_state:
                            del st.session_state['analysis_results']
                        return
                    volatility_data = returns_data.rolling(window_size).std().dropna()
                else:
                    if processed_data.shape[0] < window_size:
                        st.warning(f"Not enough data points ({processed_data.shape[0]}) for the specified rolling window size ({window_size}). Please reduce the window size or extend the date range.")
                        if 'analysis_results' in st.session_state:
                            del st.session_state['analysis_results']
                        return
                    volatility_data = processed_data.rolling(window_size).std().dropna()

                if volatility_data.empty:
                    st.error("Could not calculate rolling volatility for the selected tickers and window size.")
                    if 'analysis_results' in st.session_state:
                        del st.session_state['analysis_results']
                    return

                analysis_results['volatility_data'] = volatility_data
                # Calculate volatility correlation matrix
                volatility_corr_matrix = volatility_data.corr(method=correlation_method)
                analysis_results['volatility_corr_matrix'] = volatility_corr_matrix

            # Store all results in session state
            st.session_state['analysis_results'] = analysis_results
            st.success("Analysis complete!")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            # Clear results on error
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']


    # --- Display Results (if available in session state) ---
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        analysis_type = results['analysis_type']
        data_label = results['data_label']
        correlation_method = results['correlation_method']
        processed_data = results['processed_data'] # Retrieve processed data

        if analysis_type == 'Static Correlation':
            st.subheader(f"📊 Static Correlation Matrix ({data_label} - {correlation_method.capitalize()})")
            st.dataframe(results['corr_matrix'].style.format("{:.4f}"))

            if correlation_method == 'pearson' and 'pvalue_matrix' in results:
                st.subheader("🔬 P-value Matrix (Pearson Correlation)")
                st.markdown("P-values indicate the statistical significance of the correlation coefficient. A low p-value (e.g., < 0.05) suggests a statistically significant correlation.")
                st.dataframe(results['pvalue_matrix'].style.format("{:.4f}"))

            st.subheader("Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(results['corr_matrix'], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, annot_kws={"size": 7})
            ax.set_title(f"Correlation Matrix of {data_label} ({correlation_method.capitalize()})", fontsize=14)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
            plt.tight_layout()
            st.pyplot(fig)

            # --- Scatter Plot Section (Available for Static Correlation) ---
            if len(tickers) > 1: # Use the tickers list from the input section
                 st.markdown("---")
                 st.subheader("Scatter Plot of Ticker Pair")
                 st.markdown(f"Visualize the relationship between two tickers based on {data_label}.")

                 # Use unique keys for selectboxes to avoid conflicts with other widgets
                 scatter_ticker1 = st.selectbox(f"Select Ticker for X-axis ({data_label}):", tickers, key='scatter_ticker1_display')
                 scatter_ticker2 = st.selectbox(f"Select Ticker for Y-axis ({data_label}):", tickers, key='scatter_ticker2_display')

                 if scatter_ticker1 and scatter_ticker2 and scatter_ticker1 != scatter_ticker2:
                     fig, ax = plt.subplots(figsize=(8, 6))
                     # Use the processed_data retrieved from session state
                     ax.scatter(processed_data[scatter_ticker1], processed_data[scatter_ticker2], alpha=0.5)
                     ax.set_title(f"Scatter Plot: {scatter_ticker1} vs {scatter_ticker2} ({data_label})")
                     ax.set_xlabel(f"{scatter_ticker1} {data_label}")
                     ax.set_ylabel(f"{scatter_ticker2} {data_label}")
                     ax.grid(True)
                     st.pyplot(fig)
                 elif scatter_ticker1 == scatter_ticker2:
                     st.info("Please select two different tickers for the scatter plot.")
                 else:
                     st.info("Select a ticker pair to see the scatter plot.")


        elif analysis_type == 'Rolling Correlation':
            st.subheader(f"🔄 Rolling Correlation ({data_label} - {correlation_method.capitalize()})")
            window_size = results['window_size'] # Retrieve window size from session state
            st.markdown(f"Calculating correlation over a rolling window of **{window_size}** days.")

            st.subheader("Select Ticker Pair for Rolling Correlation Plot")
            # Use unique keys for selectboxes
            ticker1 = st.selectbox("Select first ticker:", tickers, key='roll_ticker1_display')
            ticker2 = st.selectbox("Select second ticker:", tickers, key='roll_ticker2_display')

            if ticker1 and ticker2 and ticker1 != ticker2:
                 # Recalculate rolling correlation for the selected pair using the stored processed data
                 pair_rolling_corr = calculate_rolling_correlation(
                     processed_data[ticker1],
                     processed_data[ticker2],
                     window_size,
                     correlation_method
                 )

                 if not pair_rolling_corr.empty:
                     st.subheader(f"Rolling Correlation: {ticker1} vs {ticker2}")
                     fig, ax = plt.subplots(figsize=(12, 6))
                     ax.plot(pair_rolling_corr.index, pair_rolling_corr.values)
                     ax.set_title(f"Rolling {correlation_method.capitalize()} Correlation ({window_size}-day window)")
                     ax.set_xlabel("Date")
                     ax.set_ylabel("Correlation Coefficient")
                     ax.grid(True)
                     st.pyplot(fig)
                 else:
                      st.info(f"Not enough data points to calculate rolling correlation for {ticker1} and {ticker2} with window size {window_size}. Try a smaller window or different date range.")

            elif ticker1 == ticker2:
                 st.info("Please select two different tickers for the rolling correlation plot.")
            else:
                 st.info("Select a ticker pair to see the rolling correlation plot.")


        elif analysis_type == 'Volatility Correlation':
            st.subheader(f"♨️ Volatility Correlation ({data_label})")
            window_size = results['window_size'] # Retrieve window size
            st.markdown(f"Calculating the correlation of rolling **standard deviation** of {data_label} over a **{window_size}** day window.")

            volatility_data = results['volatility_data'] # Retrieve volatility data
            volatility_corr_matrix = results['volatility_corr_matrix'] # Retrieve volatility correlation matrix

            st.subheader("Rolling Volatility Series")
            st.line_chart(volatility_data)

            st.subheader("Volatility Correlation Matrix")
            st.dataframe(volatility_corr_matrix.style.format("{:.4f}"))

            st.subheader("Volatility Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(volatility_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, annot_kws={"size": 7})
            ax.set_title(f"Volatility Correlation Matrix ({correlation_method.capitalize()})", fontsize=14)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
            plt.tight_layout()
            st.pyplot(fig)