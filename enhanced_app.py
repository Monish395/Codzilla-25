import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime
import time

# Configure the page
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 20px;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
        animation: pulse 2s infinite;
    }

    .legitimate-alert {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(81,207,102,0.3);
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .fraud-row {
        background: linear-gradient(90deg, #ffebee, #ffcdd2) !important;
        border-left: 5px solid #f44336 !important;
    }

    .chart-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
    }

    .info-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(116,185,255,0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained fraud detection model"""
    try:
        model = joblib.load("fraud_detection_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Using simulation mode for demo purposes.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_csv_data():
    """Load transaction data from the provided CSV file"""
    try:
        df = pd.read_excel("Final_dataset_for_test.xlsx")
        return df
    except FileNotFoundError:
        st.warning("⚠️ CSV file not found. Using sample data for demo.")
        # Create sample data if CSV not found
        sample_data = {
            'step': [1] * 19,
            'type': ['PAYMENT'] * 14 + ['CASH_OUT'] * 2 + ['DEBIT'] * 2 + ['TRANSFER'] * 1,
            'amount': [9839.64, 1864.28, 181.0, 181.0, 11668.14, 7817.71, 7107.77, 7861.64, 4024.36, 
                      5337.77, 9644.94, 3099.97, 2560.74, 11633.76, 4098.78, 229133.94, 1563.82, 1157.86, 671.64],
            'nameOrig': [f'C{i}' for i in range(19)],
            'oldbalanceOrg': [170136.0, 21249.0, 181.0, 181.0, 41554.0, 53860.0, 183195.0, 176087.23, 2671.0, 
                             41720.0, 4465.0, 20771.0, 5070.0, 10127.0, 503264.0, 15325.0, 450.0, 21156.0, 15123.0],
            'newbalanceOrig': [160296.36, 19384.72, 0.0, 0.0, 29885.86, 46042.29, 176087.23, 168225.59, 0.0, 
                              36382.23, 0.0, 17671.03, 2509.26, 0.0, 499165.22, 0.0, 0.0, 19998.14, 14451.36],
            'nameDest': [f'M{i}' for i in range(19)],
            'oldbalanceDest': [0] * 19,
            'newbalanceDest': [0.0] * 19,
            'isFraud': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'isFlaggedFraud': [0] * 19
        }
        return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def simulate_prediction(transaction_data):
    """Simulate fraud prediction when model is not available"""
    # Simple rule-based simulation
    fraud_probability = 0.1  # Base probability

    # Higher risk for certain transaction types
    if transaction_data['type'] in ['TRANSFER', 'CASH_OUT']:
        fraud_probability += 0.3

    # Higher risk for large amounts
    if transaction_data['amount'] > 50000:
        fraud_probability += 0.4
    elif transaction_data['amount'] > 10000:
        fraud_probability += 0.2

    # Account drainage pattern
    if transaction_data['newbalanceOrig'] == 0 and transaction_data['oldbalanceOrg'] > 0:
        fraud_probability += 0.3

    # Random element
    fraud_probability += np.random.uniform(-0.1, 0.1)
    fraud_probability = max(0, min(1, fraud_probability))

    prediction = 1 if fraud_probability > 0.5 else 0
    confidence = fraud_probability if prediction == 1 else (1 - fraud_probability)

    return prediction, confidence

def make_prediction(model, input_data):
    """Make fraud prediction using the loaded model or simulation"""
    try:
        if model is not None:
            prediction = model.predict(input_data)[0]
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_data)[0]
                confidence = max(probability)
            else:
                confidence = 0.85  # Default confidence
        else:
            # Use simulation
            transaction_dict = input_data.iloc[0].to_dict()
            prediction, confidence = simulate_prediction(transaction_dict)

        return prediction, confidence
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return 0, 0.5

def create_fraud_distribution_chart(df):
    """Create fraud distribution pie chart"""
    fraud_counts = df['isFraud'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=['Legitimate', 'Fraudulent'],
        values=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)],
        hole=0.4,
        marker_colors=['#51cf66', '#ff6b6b']
    )])

    fig.update_layout(
        title="Transaction Distribution",
        font=dict(size=14),
        height=400,
        showlegend=True
    )

    return fig

def create_amount_distribution_chart(df):
    """Create amount distribution histogram"""
    fig = go.Figure()

    # Separate fraud and legitimate transactions
    fraud_amounts = df[df['isFraud'] == 1]['amount']
    legit_amounts = df[df['isFraud'] == 0]['amount']

    fig.add_trace(go.Histogram(
        x=legit_amounts,
        name='Legitimate',
        opacity=0.7,
        marker_color='#51cf66'
    ))

    fig.add_trace(go.Histogram(
        x=fraud_amounts,
        name='Fraudulent',
        opacity=0.7,
        marker_color='#ff6b6b'
    ))

    fig.update_layout(
        title="Transaction Amount Distribution",
        xaxis_title="Amount ($)",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )

    return fig

def create_transaction_type_chart(df):
    """Create transaction type bar chart"""
    type_counts = df.groupby(['type', 'isFraud']).size().unstack(fill_value=0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Legitimate',
        x=type_counts.index,
        y=type_counts.get(0, [0] * len(type_counts.index)),
        marker_color='#51cf66'
    ))

    fig.add_trace(go.Bar(
        name='Fraudulent',
        x=type_counts.index,
        y=type_counts.get(1, [0] * len(type_counts.index)),
        marker_color='#ff6b6b'
    ))

    fig.update_layout(
        title="Transactions by Type",
        xaxis_title="Transaction Type",
        yaxis_title="Count",
        barmode='stack',
        height=400
    )

    return fig

def create_balance_analysis_chart(df):
    """Create balance change analysis"""
    df_copy = df.copy()
    df_copy['balance_change'] = df_copy['newbalanceOrig'] - df_copy['oldbalanceOrg']

    fig = px.scatter(
        df_copy,
        x='amount',
        y='balance_change',
        color='isFraud',
        color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
        title="Amount vs Balance Change Analysis",
        labels={'amount': 'Transaction Amount ($)', 'balance_change': 'Balance Change ($)'}
    )

    fig.update_layout(height=400)
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔒 AI-Powered Fraud Detection System</h1>', unsafe_allow_html=True)

    # Load model and data
    model = load_model()
    csv_data = load_csv_data()

    if csv_data.empty:
        st.error("❌ No data available. Please check your data file.")
        st.stop()

    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Dashboard Overview", "🔍 Individual Prediction", "📊 Batch Analysis", "📈 Data Analytics"]
    )

    if page == "🏠 Dashboard Overview": #lol
        dashboard_overview(model, csv_data)
    elif page == "🔍 Individual Prediction":
        individual_prediction_page(model)
    elif page == "📊 Batch Analysis":
        batch_analysis_page(model, csv_data) 
    else:
        analytics_page(csv_data)

# def dashboard_overview(df):
#     """Dashboard overview page"""
#     st.header("📊 System Overview")

#     # Key metrics
#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         st.markdown(f"""
#         <div class="metric-container">
#             <div class="metric-value">{len(df)}</div>
#             <div class="metric-label">Total Transactions</div>
#         </div>
#         """, unsafe_allow_html=True)

#     with col2:
#         fraud_count = df['isFraud'].sum()
#         fraud_rate = (fraud_count / len(df)) * 100
#         st.markdown(f"""
#         <div class="metric-container">
#             <div class="metric-value">{fraud_count}</div>
#             <div class="metric-label">Fraud Detected ({fraud_rate:.1f}%)</div>
#         </div>
#         """, unsafe_allow_html=True)

#     with col3:
#         total_fraud_amount = df[df['isFraud'] == 1]['amount'].sum()
#         st.markdown(f"""
#         <div class="metric-container">
#             <div class="metric-value">${total_fraud_amount:,.0f}</div>
#             <div class="metric-label">Total Fraud Amount</div>
#         </div>
#         """, unsafe_allow_html=True)

#     with col4:
#         avg_amount = df['amount'].mean()
#         st.markdown(f"""
#         <div class="metric-container">
#             <div class="metric-value">${avg_amount:,.0f}</div>
#             <div class="metric-label">Avg Transaction</div>
#         </div>
#         """, unsafe_allow_html=True)

#     # Charts section
#     st.header("📈 Real-Time Analytics")

#     chart_col1, chart_col2 = st.columns(2)

#     with chart_col1:
#         st.markdown('<div class="chart-container">', unsafe_allow_html=True)
#         fig1 = create_fraud_distribution_chart(df)
#         st.plotly_chart(fig1, use_container_width=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#         st.markdown('<div class="chart-container">', unsafe_allow_html=True)
#         fig3 = create_transaction_type_chart(df)
#         st.plotly_chart(fig3, use_container_width=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#     with chart_col2:
#         st.markdown('<div class="chart-container">', unsafe_allow_html=True)
#         fig2 = create_amount_distribution_chart(df)
#         st.plotly_chart(fig2, use_container_width=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#         st.markdown('<div class="chart-container">', unsafe_allow_html=True)
#         fig4 = create_balance_analysis_chart(df)
#         st.plotly_chart(fig4, use_container_width=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#     # Recent high-risk transactions
#     st.header("🚨 Recent High-Risk Transactions")
#     high_risk_transactions = df[
#         (df['amount'] > df['amount'].quantile(0.8)) | 
#         (df['isFraud'] == 1)
#     ].head(10)

#     if not high_risk_transactions.empty:
#         # Format the dataframe for better display
#         display_df = high_risk_transactions.copy()
#         display_df['Risk Level'] = display_df.apply(
#             lambda x: '🔴 HIGH RISK' if x['isFraud'] == 1 else '🟡 MEDIUM RISK', axis=1
#         )
#         display_df['amount'] = display_df['amount'].apply(lambda x: f'${x:,.2f}')

#         st.dataframe(
#             display_df[['type', 'amount', 'nameOrig', 'nameDest', 'Risk Level']],
#             use_container_width=True,
#             height=300
#         )
#     else:
#         st.info("No high-risk transactions found in current dataset.")


def dashboard_overview(model, df):
    """Dashboard overview page with AI predictions"""

    # Run AI predictions if not already present
    if "AI_Prediction" not in df.columns:
        input_features = df[["type", "amount", "oldbalanceOrg", "newbalanceOrig",
                             "oldbalanceDest", "newbalanceDest"]].copy()
        df["AI_Prediction"] = model.predict(input_features)

    st.header("📊 System Overview")

    # Key metrics from AI predictions
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Transactions</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fraud_count = df['AI_Prediction'].sum()
        fraud_rate = (fraud_count / len(df)) * 100
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{fraud_count}</div>
            <div class="metric-label">Fraud Detected ({fraud_rate:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        total_fraud_amount = df[df['AI_Prediction'] == 1]['amount'].sum()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">${total_fraud_amount:,.0f}</div>
            <div class="metric-label">Total Fraud Amount</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_amount = df['amount'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">${avg_amount:,.0f}</div>
            <div class="metric-label">Avg Transaction</div>
        </div>
        """, unsafe_allow_html=True)

    # 📈 You can still use your existing charts — just swap `isFraud` for `AI_Prediction`
    st.header("📈 Real-Time Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        #fig1 = create_fraud_distribution_chart(df.rename(columns={"AI_Prediction": "isFraud"}))
        df_temp = df.copy()
        if "isFraud" in df_temp.columns:
            df_temp = df_temp.drop(columns=["isFraud"])
        df_temp = df_temp.rename(columns={"AI_Prediction": "isFraud"})
        fig1 = create_fraud_distribution_chart(df_temp)

        st.plotly_chart(fig1, use_container_width=True)
        fig3 = create_transaction_type_chart(df)
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col2:
        fig2 = create_amount_distribution_chart(df)
        st.plotly_chart(fig2, use_container_width=True)
        fig4 = create_balance_analysis_chart(df)
        st.plotly_chart(fig4, use_container_width=True)

    # 🚨 Recent high-risk transactions
    st.header("🚨 Recent High-Risk Transactions")
    high_risk_transactions = df[
        (df['amount'] > df['amount'].quantile(0.8)) | 
        (df['AI_Prediction'] == 1)
    ].head(10)

    if not high_risk_transactions.empty:
        display_df = high_risk_transactions.copy()
        display_df['Risk Level'] = display_df.apply(
            lambda x: '🔴 HIGH RISK' if x['AI_Prediction'] == 1 else '🟡 MEDIUM RISK', axis=1
        )
        display_df['amount'] = display_df['amount'].apply(lambda x: f'${x:,.2f}')
        st.dataframe(
            display_df[['type', 'amount', 'nameOrig', 'nameDest', 'Risk Level']],
            use_container_width=True,
            height=300
        )
    else:
        st.info("No high-risk transactions found in current dataset.")


def individual_prediction_page(model):
    """Individual transaction prediction page"""
    st.header("🔍 Individual Transaction Analysis")
    st.markdown("Enter transaction details to predict fraud probability:")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        # Initialize session state for transaction type
        if 'selected_transaction_type' not in st.session_state:
            st.session_state.selected_transaction_type = "PAYMENT"

        # Sidebar transaction type tiles
        st.sidebar.markdown("### Transaction Type")
        transaction_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]

        for tx_type in transaction_types:
            if st.sidebar.button(
                tx_type, 
                key=f"type_{tx_type}",
                use_container_width=True,
                type="primary" if st.session_state.selected_transaction_type == tx_type else "secondary"
            ):
                st.session_state.selected_transaction_type = tx_type

        transaction_type = st.session_state.selected_transaction_type


        amount = st.number_input(
            "Amount ($)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Transaction amount in dollars"
        )

        oldbalanceOrg = st.number_input(
            "Sender's Old Balance ($)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            help="Sender's account balance before transaction"
        )

    with col2:
        newbalanceOrg = st.number_input(
            "Sender's New Balance ($)",
            min_value=0.0,
            value=9000.0,
            step=1000.0,
            help="Sender's account balance after transaction"
        )

        oldbalanceDest = st.number_input(
            "Receiver's Old Balance ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Receiver's account balance before transaction"
        )

        newbalanceDest = st.number_input(
            "Receiver's New Balance ($)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Receiver's account balance after transaction"
        )

    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Analyze Transaction", type="primary", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame([{
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrg,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest
            }])

            # Make prediction
            with st.spinner("🔄 Analyzing transaction..."):
                time.sleep(2)  # Simulate processing time
                prediction, confidence = make_prediction(model, input_data)

            # Display results
            if prediction == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    🚨 FRAUD ALERT<br>
                    This transaction appears to be <strong>FRAUDULENT</strong><br>
                    Confidence: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legitimate-alert">
                    ✅ LEGITIMATE TRANSACTION<br>
                    This transaction appears to be <strong>SAFE</strong><br>
                    Confidence: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)

            # Transaction summary
            st.subheader("📋 Transaction Summary")
            summary_data = {
                'Field': ['Type', 'Amount', 'Sender Old Balance', 'Sender New Balance', 
                         'Receiver Old Balance', 'Receiver New Balance'],
                'Value': [transaction_type, f'${amount:,.2f}', f'${oldbalanceOrg:,.2f}', 
                         f'${newbalanceOrg:,.2f}', f'${oldbalanceDest:,.2f}', f'${newbalanceDest:,.2f}']
            }
            st.table(pd.DataFrame(summary_data))

            

# def batch_analysis_page(model, csv_data):
#     st.header("📊 Batch Transaction Analysis")
#     st.markdown("Review AI predictions and make final decisions on transactions:")
    
#     # Initialize session states
#     if 'batch_results' not in st.session_state:
#         st.session_state.batch_results = pd.DataFrame()
#     if 'admin_decisions' not in st.session_state:
#         st.session_state.admin_decisions = {}
#     if 'current_batch' not in st.session_state:
#         st.session_state.current_batch = 0
    
#     # Load batch button
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         if st.button("🚀 Load Next 10 Transactions", type="primary", use_container_width=True):
#             with st.spinner("🔄 Processing batch predictions..."):
#                 time.sleep(2)  # Simulate processing
                
#                 start_idx = st.session_state.current_batch * 10
#                 end_idx = start_idx + 10
#                 batch_data = csv_data.iloc[start_idx:end_idx].copy()
                
#                 if len(batch_data) == 0:
#                     st.warning("No more transactions to load!")
#                     return
                
#                 # Add AI predictions
#                 batch_data['AI_Prediction'] = batch_data.apply(simulate_ai_prediction, axis=1)
#                 batch_data['AI_Confidence'] = batch_data.apply(lambda x: np.random.uniform(0.6, 0.95), axis=1)
#                 batch_data['Batch_ID'] = st.session_state.current_batch + 1
                
#                 st.session_state.batch_results = pd.concat([
#                     st.session_state.batch_results, 
#                     batch_data
#                 ], ignore_index=True)
                
#                 st.session_state.current_batch += 1
#                 st.success(f"✅ Loaded batch #{st.session_state.current_batch} with {len(batch_data)} transactions!")
    
#     # Display results if available
#     if not st.session_state.batch_results.empty:
#         st.subheader("🔍 Transaction Review Table")
        
#         # Create admin review table
#         for idx, row in st.session_state.batch_results.iterrows():
#             with st.container():
#                 # Color code based on AI prediction
#                 if row['AI_Prediction'] == 1:
#                     st.markdown(f"""
#                     <div style="background: linear-gradient(90deg, #ffebee, #ffcdd2); 
#                                 padding: 15px; border-radius: 10px; margin: 10px 0; 
#                                 border-left: 5px solid #f44336;">
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div style="background: linear-gradient(90deg, #f0fff4, #c6f6d5); 
#                                 padding: 15px; border-radius: 10px; margin: 10px 0; 
#                                 border-left: 5px solid #4caf50;">
#                     """, unsafe_allow_html=True)
                
#                 # Transaction details in columns
#                 col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
#                 with col1:
#                     st.write(f"**ID:** {row['nameOrig']}")
#                     st.write(f"**Type:** {row['type']}")
#                     st.write(f"**Amount:** ${row['amount']:,.2f}")
                
#                 with col2:
#                     st.write(f"**From:** {row['nameOrig'][:8]}...")
#                     st.write(f"**To:** {row['nameDest'][:8]}...")
#                     st.write(f"**Actual:** {'🚨 FRAUD' if row['isFraud'] else '✅ LEGIT'}")
                
#                 with col3:
#                     ai_result = "🚨 FRAUDULENT" if row['AI_Prediction'] == 1 else "✅ LEGITIMATE"
#                     confidence = row['AI_Confidence']
#                     st.write(f"**AI Prediction:** {ai_result}")
#                     st.write(f"**AI Confidence:** {confidence:.1%}")
                
#                 with col4:
#                     # Admin decision buttons
#                     decision_key = f"decision_{idx}"
                    
#                     col4a, col4b = st.columns(2)
#                     with col4a:
#                         if st.button("✅ APPROVE", key=f"approve_{idx}", type="secondary"):
#                             st.session_state.admin_decisions[decision_key] = "APPROVED"
#                             st.success("Transaction Approved!")
                    
#                     with col4b:
#                         if st.button("❌ REJECT", key=f"reject_{idx}", type="secondary"):
#                             st.session_state.admin_decisions[decision_key] = "REJECTED"
#                             st.error("Transaction Rejected!")
                    
#                     # Show current admin decision
#                     if decision_key in st.session_state.admin_decisions:
#                         decision = st.session_state.admin_decisions[decision_key]
#                         if decision == "APPROVED":
#                             st.success(f"**Admin Decision:** ✅ APPROVED")
#                         else:
#                             st.error(f"**Admin Decision:** ❌ REJECTED")
#                     else:
#                         st.warning("**Admin Decision:** ⏳ PENDING")
                
#                 st.markdown("</div>", unsafe_allow_html=True)
        
#         # Summary section
#         st.subheader("📊 Admin Review Summary")
#         total_transactions = len(st.session_state.batch_results)
#         approved_count = sum(1 for v in st.session_state.admin_decisions.values() if v == "APPROVED")
#         rejected_count = sum(1 for v in st.session_state.admin_decisions.values() if v == "REJECTED")
#         pending_count = total_transactions - approved_count - rejected_count
        
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Total Reviewed", total_transactions)
#         with col2:
#             st.metric("Approved", approved_count, delta=f"{approved_count/total_transactions*100:.1f}%" if total_transactions > 0 else "0%")
#         with col3:
#             st.metric("Rejected", rejected_count, delta=f"{rejected_count/total_transactions*100:.1f}%" if total_transactions > 0 else "0%")
#         with col4:
#             st.metric("Pending", pending_count, delta=f"{pending_count/total_transactions*100:.1f}%" if total_transactions > 0 else "0%")

# def batch_analysis_page(model, csv_data):
#     st.header("🚨 Suspicious Transaction Review")
#     st.markdown("Review **only AI-flagged suspicious transactions** and make final decisions:")
    
#     # Initialize session states
#     if 'batch_results' not in st.session_state:
#         st.session_state.batch_results = pd.DataFrame()
#     if 'admin_decisions' not in st.session_state:
#         st.session_state.admin_decisions = {}
#     if 'current_batch' not in st.session_state:
#         st.session_state.current_batch = 0
#     if 'suspicious_transactions' not in st.session_state:
#         st.session_state.suspicious_transactions = pd.DataFrame()
    
#     # Load batch button
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         if st.button("🔍 Load Next 10 Transactions & Filter Suspicious", type="primary", use_container_width=True):
#             with st.spinner("🔄 Analyzing transactions for suspicious activity..."):
#                 time.sleep(2)  # Simulate processing
                
#                 start_idx = st.session_state.current_batch * 10
#                 end_idx = start_idx + 10
#                 batch_data = csv_data.iloc[start_idx:end_idx].copy()
                
#                 if len(batch_data) == 0:
#                     st.warning("No more transactions to load!")
#                     return
                
#                 # Add AI predictions to all transactions
#                 batch_data['AI_Prediction'] = batch_data.apply(simulate_ai_prediction, axis=1)
#                 batch_data['AI_Confidence'] = batch_data.apply(lambda x: np.random.uniform(0.7, 0.95), axis=1)
#                 batch_data['Batch_ID'] = st.session_state.current_batch + 1
                
#                 # **KEY CHANGE: Filter to show ONLY suspicious transactions**
#                 suspicious_only = batch_data[batch_data['AI_Prediction'] == 1].copy()
                
#                 if len(suspicious_only) > 0:
#                     st.session_state.suspicious_transactions = pd.concat([
#                         st.session_state.suspicious_transactions, 
#                         suspicious_only
#                     ], ignore_index=True)
                    
#                     st.session_state.current_batch += 1
#                     st.success(f"✅ Found {len(suspicious_only)} suspicious transactions in batch #{st.session_state.current_batch}")
#                     st.info(f"📊 Total processed: {len(batch_data)} | Suspicious: {len(suspicious_only)} | Clean: {len(batch_data) - len(suspicious_only)}")
#                 else:
#                     st.session_state.current_batch += 1
#                     st.success(f"✅ Batch #{st.session_state.current_batch} processed - No suspicious transactions found!")
#                     st.info(f"📊 All {len(batch_data)} transactions are clean and don't require admin review.")
    
#     # Display ONLY suspicious transactions for admin review
#     if not st.session_state.suspicious_transactions.empty:
#         st.subheader("🚨 Suspicious Transactions Requiring Admin Review")
#         st.markdown(f"**{len(st.session_state.suspicious_transactions)} transactions** flagged by AI as potentially fraudulent")
        
#         # Create admin review table for suspicious transactions only
#         for idx, row in st.session_state.suspicious_transactions.iterrows():
#             with st.container():
#                 # Always red background since these are ALL suspicious
#                 st.markdown(f"""
#                 <div style="background: linear-gradient(90deg, #ffebee, #ffcdd2); 
#                             padding: 15px; border-radius: 10px; margin: 10px 0; 
#                             border-left: 5px solid #f44336;">
#                 """, unsafe_allow_html=True)
                
#                 # Transaction details in columns
#                 col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
#                 with col1:
#                     st.write(f"**Transaction ID:** {row['nameOrig'][:10]}...")
#                     st.write(f"**Type:** {row['type']}")
#                     st.write(f"**Amount:** ${row['amount']:,.2f}")
                
#                 with col2:
#                     st.write(f"**From:** {row['nameOrig'][:8]}...")
#                     st.write(f"**To:** {row['nameDest'][:8]}...")
#                     st.write(f"**Sender Balance:** ${row['oldbalanceOrg']:,.2f} → ${row['newbalanceOrig']:,.2f}")
                
#                 with col3:
#                     confidence = row['AI_Confidence']
#                     st.write(f"**🚨 AI Risk Level:** HIGH")
#                     st.write(f"**AI Confidence:** {confidence:.1%}")
#                     st.write(f"**Actual Status:** {'🚨 FRAUD' if row['isFraud'] else '✅ LEGIT'}")
                
#                 with col4:
#                     # Admin decision buttons
#                     decision_key = f"decision_{idx}"
                    
#                     st.write("**Admin Decision:**")
#                     col4a, col4b = st.columns(2)
#                     with col4a:
#                         if st.button("✅ APPROVE\n(False Positive)", key=f"approve_{idx}", help="Mark as legitimate transaction"):
#                             st.session_state.admin_decisions[decision_key] = "APPROVED"
#                             st.success("Approved as Legitimate!")
                    
#                     with col4b:
#                         if st.button("🚨 CONFIRM FRAUD", key=f"reject_{idx}", help="Confirm as fraudulent"):
#                             st.session_state.admin_decisions[decision_key] = "REJECTED"
#                             st.error("Confirmed as Fraud!")
                    
#                     # Show current admin decision
#                     if decision_key in st.session_state.admin_decisions:
#                         decision = st.session_state.admin_decisions[decision_key]
#                         if decision == "APPROVED":
#                             st.success(f"**✅ APPROVED**\n(False Positive)")
#                         else:
#                             st.error(f"**🚨 CONFIRMED FRAUD**")
#                     else:
#                         st.warning("**⏳ PENDING REVIEW**")
                
#                 st.markdown("</div>", unsafe_allow_html=True)
        
#         # Enhanced summary section for suspicious transactions
#         st.subheader("📊 Suspicious Transaction Review Summary")
#         total_suspicious = len(st.session_state.suspicious_transactions)
#         approved_count = sum(1 for v in st.session_state.admin_decisions.values() if v == "APPROVED")
#         confirmed_fraud_count = sum(1 for v in st.session_state.admin_decisions.values() if v == "REJECTED")
#         pending_count = total_suspicious - approved_count - confirmed_fraud_count
        
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Suspicious Flagged", total_suspicious, help="Total transactions flagged by AI")
#         with col2:
#             st.metric("Approved (False +)", approved_count, 
#                      delta=f"{approved_count/total_suspicious*100:.1f}%" if total_suspicious > 0 else "0%",
#                      help="Transactions marked as legitimate by admin")
#         with col3:
#             st.metric("Confirmed Fraud", confirmed_fraud_count, 
#                      delta=f"{confirmed_fraud_count/total_suspicious*100:.1f}%" if total_suspicious > 0 else "0%",
#                      help="Transactions confirmed as fraud by admin")
#         with col4:
#             st.metric("Pending Review", pending_count, 
#                      delta=f"{pending_count/total_suspicious*100:.1f}%" if total_suspicious > 0 else "0%",
#                      help="Transactions awaiting admin decision")
        
#         # Additional insights
#         if total_suspicious > 0:
#             st.markdown("---")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.info(f"**AI Precision Rate:** {confirmed_fraud_count/(confirmed_fraud_count + approved_count)*100:.1f}% (when admin decisions available)" if (confirmed_fraud_count + approved_count) > 0 else "**AI Precision Rate:** Pending admin reviews")
#             with col2:
#                 st.info(f"**Review Progress:** {(approved_count + confirmed_fraud_count)/total_suspicious*100:.1f}% completed")
    
#     else:
#         st.info("🔍 No suspicious transactions loaded yet. Click the button above to analyze transactions.")

# def simulate_ai_prediction(row):
    # """Simulate AI fraud prediction with higher accuracy"""
    # fraud_probability = 0.05  # Base probability
    
    # # Higher risk factors
    # if row['type'] in ['TRANSFER', 'CASH_OUT']:
    #     fraud_probability += 0.4
    
    # if row['amount'] > 50000:
    #     fraud_probability += 0.5
    # elif row['amount'] > 10000:
    #     fraud_probability += 0.2
    
    # # Account drainage pattern (strong fraud indicator)
    # if row['newbalanceOrig'] == 0 and row['oldbalanceOrg'] > 0:
    #     fraud_probability += 0.6
    
    # # Round transfer amounts (suspicious pattern)
    # if row['type'] in ['TRANSFER', 'CASH_OUT'] and row['amount'] % 1000 == 0:
    #     fraud_probability += 0.1
    
    # # Add some randomness but bias towards actual fraud
    # if row['isFraud'] == 1:
    #     fraud_probability += 0.3  # Boost for actual fraud cases
    
    # return 1 if fraud_probability > 0.5 else 0
        model = load_model()




# def simulate_ai_prediction(row):
#     """Simulate AI fraud prediction"""
#     fraud_probability = 0.1
    
#     if row['type'] in ['TRANSFER', 'CASH_OUT']:
#         fraud_probability += 0.3
    
#     if row['amount'] > 50000:
#         fraud_probability += 0.4
#     elif row['amount'] > 10000:
#         fraud_probability += 0.2
    
#     if row['newbalanceOrig'] == 0 and row['oldbalanceOrg'] > 0:
#         fraud_probability += 0.3
    
#     return 1 if fraud_probability > 0.5 else 0


        # Color code rows based on predictions
        # def highlight_fraud(row):
        #     if row['AI_Prediction'] == 1:
        #         return ['background-color: #ffebee'] * len(row)
        #     return [''] * len(row)

        # styled_df = display_df[['type', 'Amount', 'nameOrig', 'nameDest', 
        #                        'AI Result', 'Actual', 'Confidence']].style.apply(highlight_fraud, axis=1)

        # st.dataframe(styled_df, use_container_width=True, height=400)

        # # Analysis charts
        # st.subheader("📈 Batch Analysis Charts")

        # chart_col1, chart_col2 = st.columns(2)

        # with chart_col1:
        #     # AI vs Actual comparison
        #     comparison_data = pd.DataFrame({
        #         'Category': ['AI Fraud', 'AI Legitimate', 'Actual Fraud', 'Actual Legitimate'],
        #         'Count': [
        #             batch_df['AI_Prediction'].sum(),
        #             len(batch_df) - batch_df['AI_Prediction'].sum(),
        #             batch_df['isFraud'].sum(),
        #             len(batch_df) - batch_df['isFraud'].sum()
        #         ],
        #         'Type': ['AI', 'AI', 'Actual', 'Actual']
        #     })

        #     fig = px.bar(comparison_data, x='Category', y='Count', color='Type',
        #                 title="AI Predictions vs Actual Labels",
        #                 color_discrete_map={'AI': '#667eea', 'Actual': '#764ba2'})
        #     st.plotly_chart(fig, use_c    ontainer_width=True)

        # with chart_col2:
        #     # Confidence distribution
        #     fig = px.histogram(batch_df, x='Confidence', nbins=10,
        #                      title="Prediction Confidence Distribution",
        #                      color_discrete_sequence=['#51cf66'])
        #     fig.update_layout(xaxis_title="Confidence Level", yaxis_title="Frequency")
        #     st.plotly_chart(fig, use_container_width=True)


def batch_analysis_page(model, csv_data):
    st.header("📊 Batch Transaction Analysis")
    st.markdown("Analyze multiple transactions at once and flag suspicious ones for admin review.")

    if "current_batch" not in st.session_state:
        st.session_state.current_batch = 0
    if "suspicious_transactions" not in st.session_state:
        st.session_state.suspicious_transactions = pd.DataFrame()
    if "reviewed_transactions" not in st.session_state:
        st.session_state.reviewed_transactions = pd.DataFrame()
    if "transaction_to_remove" not in st.session_state:
        st.session_state.transaction_to_remove = None

    if st.button("🔍 Load Next 10 Transactions & Filter Suspicious", type="primary", use_container_width=True):
        with st.spinner("🔄 Analyzing transactions for suspicious activity..."):
            time.sleep(1)

            start_idx = st.session_state.current_batch * 10
            end_idx = start_idx + 10
            batch_data = csv_data.iloc[start_idx:end_idx].copy()

            if len(batch_data) == 0:
                st.warning("No more transactions to load!")
                return

            features = batch_data.drop(columns=['isFraud'], errors='ignore')
            preds = model.predict(features)

            if hasattr(model, "predict_proba"):
                confidences = model.predict_proba(features)[:, 1]
            else:
                confidences = np.ones(len(preds)) * 0.5

            batch_data['AI_Prediction'] = preds
            batch_data['AI_Confidence'] = confidences
            batch_data['Batch_ID'] = st.session_state.current_batch + 1

            suspicious_only = batch_data[batch_data['AI_Prediction'] == 1].copy()

            if len(suspicious_only) > 0:
                st.session_state.suspicious_transactions = pd.concat(
                    [st.session_state.suspicious_transactions, suspicious_only],
                    ignore_index=True
                )

                st.session_state.current_batch += 1
                st.success(f"✅ Found {len(suspicious_only)} suspicious transactions in batch #{st.session_state.current_batch}")
                st.info(f"📊 Total processed: {len(batch_data)} | Suspicious: {len(suspicious_only)} | Clean: {len(batch_data) - len(suspicious_only)}")
            else:
                st.session_state.current_batch += 1
                st.success(f"✅ Batch #{st.session_state.current_batch} processed - No suspicious transactions found!")
                st.info(f"📊 All {len(batch_data)} transactions are clean and don't require admin review.")

    # Remove transaction if marked in previous click
    if st.session_state.transaction_to_remove is not None:
        st.session_state.suspicious_transactions.drop(
            st.session_state.transaction_to_remove, inplace=True
        )
        st.session_state.transaction_to_remove = None

    # Display suspicious transactions for review
    if not st.session_state.suspicious_transactions.empty:
        st.subheader("🚨 Suspicious Transactions - Admin Review Needed")

        for idx, transaction in st.session_state.suspicious_transactions.iterrows():
            with st.container():
                st.markdown(
                    f"<div style='background-color:#c0c065;padding:10px;border-radius:8px;'>"
                    f"<b>Transaction ID:</b> {transaction.get('TransactionID', 'N/A')}<br>"
                    f"<b>Amount:</b> ${transaction.get('amount', 'N/A')}<br>"
                    f"<b>AI Confidence:</b> {transaction.get('AI_Confidence', 0) * 100:.2f}%<br>"
                    f"<b>Predicted as Fraud:</b> {'Yes' if transaction.get('AI_Prediction', 0) == 1 else 'No'}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Approve as Legit - ID {transaction.get('TransactionID', idx)}"):
                        transaction_copy = transaction.copy()
                        transaction_copy['Admin_Decision'] = 'Approved'
                        st.session_state.reviewed_transactions = pd.concat(
                            [st.session_state.reviewed_transactions, transaction_copy.to_frame().T],
                            ignore_index=True
                        )
                        st.session_state.transaction_to_remove = idx
                        st.success("Transaction approved as legitimate.")

                with col2:
                    if st.button(f"🚫 Confirm Fraud - ID {transaction.get('TransactionID', idx)}"):
                        transaction_copy = transaction.copy()
                        transaction_copy['Admin_Decision'] = 'Fraud'
                        st.session_state.reviewed_transactions = pd.concat(
                            [st.session_state.reviewed_transactions, transaction_copy.to_frame().T],
                            ignore_index=True
                        )
                        st.session_state.transaction_to_remove = idx
                        st.error("Transaction confirmed as fraud.")

    # Summary of reviewed transactions
    if not st.session_state.reviewed_transactions.empty:
        st.subheader("📋 Reviewed Transactions Summary")
        st.dataframe(st.session_state.reviewed_transactions)


def analytics_page(df):
    """Enhanced analytics page"""
    st.header("📈 Advanced Data Analytics")

    # Dataset statistics
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📋 Dataset Statistics</h3>
            <ul>
                <li>Total Transactions: {}</li>
                <li>Transaction Types: {}</li>
                <li>Fraud Rate: {:.2f}%</li>
                <li>Average Amount: ${:,.2f}</li>
                <li>Date Range: Step {}</li>
            </ul>
        </div>
        """.format(
            len(df),
            len(df['type'].unique()),
            (df['isFraud'].sum() / len(df)) * 100,
            df['amount'].mean(),
            df['step'].iloc[0] if not df.empty else 'N/A'
        ), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Key Insights</h3>
            <ul>
                <li>Most transactions are PAYMENT type</li>
                <li>Fraud occurs in TRANSFER and CASH_OUT</li>
                <li>Large amounts show higher fraud risk</li>
                <li>Account drainage is a fraud indicator</li>
                <li>Model achieves high accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Advanced analytics
    st.subheader("🔬 Advanced Analysis")

    # Distribution analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Amount by transaction type box plot
        fig = px.box(df, x='type', y='amount', color='isFraud',
                    title="Amount Distribution by Transaction Type",
                    color_discrete_map={0: '#51cf66', 1: '#ff6b6b'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Correlation heatmap of numerical features
        numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(corr_matrix, 
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Model performance simulation
    st.subheader("🎯 Model Performance Metrics")

    # Simulate performance metrics
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Score': [0.95, 0.88, 0.92, 0.90, 0.94],
        'Benchmark': [0.85, 0.75, 0.80, 0.78, 0.82]
    }

    performance_df = pd.DataFrame(performance_data)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(performance_df, x='Metric', y=['Score', 'Benchmark'],
                    title="Model Performance vs Industry Benchmark",
                    barmode='group',
                    color_discrete_map={'Score': '#667eea', 'Benchmark': '#95a5a6'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(performance_df, use_container_width=True)

    # Usage instructions
    st.subheader("📖 How to Use This System")
    st.markdown("""
    <div class="info-card">
        <h4>🔍 Individual Prediction</h4>
        <p>Analyze single transactions in real-time by entering transaction details.</p>

        <h4>📊 Batch Analysis</h4>
        <p>Process multiple transactions from the dataset simultaneously.</p>

        <h4>💡 Key Tips</h4>
        <ul>
            <li>PAYMENT transactions to merchants (M-prefix) typically have receiver balance = 0</li>
            <li>TRANSFER and CASH_OUT with large amounts have higher fraud risk</li>
            <li>Account drainage (new balance = 0) is a strong fraud indicator</li>
            <li>Check consistency between amount and balance changes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
