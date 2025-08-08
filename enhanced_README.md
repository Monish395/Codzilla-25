# 🔒 AI-Powered Fraud Detection System

A comprehensive fraud detection web application built with Streamlit that uses machine learning to identify fraudulent financial transactions. Features embedded charts, real-time analytics, and batch processing capabilities.

## ✨ Features

### 🏠 Dashboard Overview
- **Real-time metrics** with animated counters
- **Interactive charts** embedded in the interface:
  - Fraud distribution pie chart
  - Transaction amount histogram
  - Transaction type analysis
  - Balance change scatter plot
- **High-risk transaction alerts**
- **Professional gradient styling**

### 🔍 Individual Prediction
- **Real-time fraud analysis** for single transactions
- **Interactive form** with validation
- **Animated prediction results** with confidence scores
- **Transaction summary** with formatted display

### 📊 Batch Analysis
- **CSV data integration** from your Excel file
- **10-transaction batch processing** from dataset
- **AI vs Actual comparison** charts
- **Confidence distribution** analysis
- **Fraud highlighting** in results table

### 📈 Advanced Analytics
- **Correlation heatmaps** for feature analysis
- **Box plots** for amount distribution
- **Performance metrics** visualization
- **Dataset statistics** and insights

## 🛠️ Installation & Setup

### 1. Prerequisites
```bash
python -m venv fraud_env
source fraud_env/bin/activate  # Windows: fraud_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. File Structure Setup
```
fraud-detection-system/
│
├── enhanced_app.py                 # Main Streamlit application ⭐
├── fraud_detection_pipeline.pkl    # Your trained ML model
├── Minimal_Data_For_Hackathon.xlsx # Transaction dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
└── .streamlit/
    └── config.toml                # Streamlit configuration
```

### 4. Add Your Files
- **Model**: Place `fraud_detection_pipeline.pkl` in the root directory
- **Data**: Place `Minimal_Data_For_Hackathon.xlsx` in the root directory

### 5. Run the Application
```bash
streamlit run enhanced_app.py
```

## 📊 Data Integration

The app automatically loads your Excel file with the following columns:
- `step`: Transaction step/timestamp
- `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT)
- `amount`: Transaction amount
- `nameOrig`: Sender account ID
- `oldbalanceOrg`: Sender's old balance
- `newbalanceOrig`: Sender's new balance
- `nameDest`: Receiver account ID
- `oldbalanceDest`: Receiver's old balance
- `newbalanceDest`: Receiver's new balance
- `isFraud`: Actual fraud label (0/1)
- `isFlaggedFraud`: System flag (0/1)

## 🎨 Visual Features

### Enhanced UI Elements
- **Gradient backgrounds** and animated components
- **Professional color scheme** with brand consistency
- **Responsive design** for desktop and mobile
- **Loading animations** and progress indicators
- **Interactive tooltips** and help text

### Embedded Charts
- **Plotly integration** for interactive visualizations
- **Real-time updates** based on data changes
- **Professional styling** matching app theme
- **Export capabilities** for charts and data

## 🔧 Model Integration

### Compatible Model Format
Your model should be saved using joblib and support:

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("fraud_detection_pipeline.pkl")

# Input format
input_data = pd.DataFrame([{
    "type": "PAYMENT",
    "amount": 1000.0,
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 9000.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}])

# Prediction
prediction = model.predict(input_data)[0]  # Returns 0 or 1
probabilities = model.predict_proba(input_data)[0]  # Optional
```

### Fallback Mode
If model file is not found, the app runs in simulation mode with:
- **Rule-based predictions** for demonstration
- **Realistic confidence scores** based on transaction patterns
- **Educational fraud indicators** for learning purposes

## 🚀 Usage Guide

### Dashboard Overview
1. **Launch the app** and view real-time metrics
2. **Explore interactive charts** showing fraud patterns  
3. **Monitor high-risk transactions** in the alert section
4. **Navigate between pages** using the sidebar

### Individual Prediction
1. **Enter transaction details** in the form
2. **Click "Analyze Transaction"** for real-time prediction
3. **View animated results** with confidence scores
4. **Review transaction summary** for verification

### Batch Analysis
1. **Click "Load Next 10 Transactions"** to process data
2. **View comparative analysis** of AI vs actual labels
3. **Explore confidence distributions** and patterns
4. **Review detailed transaction table** with fraud highlighting

## 📈 Advanced Features

### Performance Monitoring
- **Real-time accuracy tracking**
- **Confidence score analysis**
- **False positive/negative monitoring**
- **Performance benchmarking**

### Data Insights
- **Fraud pattern identification**
- **Transaction type analysis**  
- **Amount threshold insights**
- **Balance change indicators**

## 🎯 Customization

### Styling Customization
Modify the CSS in the app file to change:
- **Color schemes** and gradients
- **Animation speeds** and effects
- **Layout proportions** and spacing
- **Chart themes** and styling

### Feature Extensions
Easy to add:
- **Additional data sources**
- **New visualization types**
- **Export functionality**
- **User authentication**
- **Real-time alerts**

## 📊 Performance

### Optimizations Included
- **Streamlit caching** for data and model loading
- **Efficient data processing** with pandas
- **Optimized chart rendering** with Plotly
- **Memory management** for large datasets

### Scalability
- **Batch processing** for large datasets
- **Incremental loading** for performance
- **Modular architecture** for easy extension
- **Database integration** ready

## 🔒 Security

### Data Protection
- **No external data transmission**
- **Local processing** only
- **Secure model loading**
- **Input validation** and sanitization

## 🐛 Troubleshooting

### Common Issues
1. **Model loading errors**: Check file path and joblib compatibility
2. **Data loading issues**: Verify Excel file format and columns
3. **Chart rendering problems**: Clear browser cache and refresh
4. **Performance issues**: Reduce batch size or optimize data

### Support
- **Error logging** with detailed messages
- **Fallback modes** for missing components
- **User-friendly error displays**
- **Debug information** in development mode

## 📝 License

MIT License - Free for personal and commercial use.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For issues, questions, or feature requests:
- **Open a GitHub issue**
- **Check documentation**
- **Review troubleshooting guide**
- **Contact development team**

---

**Built with ❤️ for fraud detection professionals**
