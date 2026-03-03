# app.py - Enhanced Sustainable Textile Demand Forecasting App with Profitability Analysis
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌱 Sustainable Textile Forecasting",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cached functions for performance
@st.cache_resource
def load_model():
    """Cached model loading"""
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, "Model file not found. Please ensure model.pkl exists."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Universal file format detection and processing
@st.cache_data
def detect_and_process_file(uploaded_file):
    """Automatically detect and process any textile-related file format"""
    
    # Read file based on extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            st.error(f"❌ Unsupported file format: {file_extension}")
            return None
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None
    
    if df.empty:
        st.warning("⚠️ The uploaded file is empty")
        return None
    
    # Auto-detect and standardize column names
    df = standardize_columns(df)
    
    # Auto-detect date columns and parse them
    df = detect_and_parse_dates(df)
    
    return df

@st.cache_data
def standardize_columns(df):
    """Automatically detect and standardize column names"""
    
    # Create mapping of possible column names to standard names
    column_mappings = {
        'date': ['date', 'Date', 'DATE', 'order_date', 'Order_Date', 'transaction_date', 'sale_date', 'planned_delivery', 'actual_delivery'],
        'sku': ['sku', 'SKU', 'product_id', 'Product_ID', 'item_code', 'Item_Code', 'product_code', 'product_type'],
        'store': ['store', 'Store', 'STORE', 'location', 'Location', 'shop', 'branch', 'outlet', 'supplier'],
        'units': ['units', 'Units', 'UNITS', 'quantity', 'Quantity', 'qty', 'sold', 'sales_qty', 'order_quantity'],
        'material': ['material', 'Material', 'MATERIAL', 'fabric', 'Fabric', 'fiber', 'textile_type'],
        'category': ['category', 'Category', 'CATEGORY', 'product_type', 'type', 'class', 'segment'],
        'price': ['price', 'Price', 'PRICE', 'unit_price', 'cost', 'amount', 'value', 'cost_per_unit'],
        'revenue': ['revenue', 'Revenue', 'REVENUE', 'total_sales', 'sales_amount', 'total_amount', 'total_order_cost']
    }
    
    # Apply column mappings
    column_renames = {}
    for standard_name, possible_names in column_mappings.items():
        for possible_name in possible_names:
            if possible_name in df.columns:
                column_renames[possible_name] = standard_name
                break
    
    # Rename columns
    df = df.rename(columns=column_renames)
    
    return df

@st.cache_data
def detect_and_parse_dates(df):
    """Automatically detect and parse date columns"""
    
    date_columns = ['date', 'order_date', 'transaction_date', 'sale_date', 'planned_delivery', 'actual_delivery']
    
    for col in df.columns:
        if col.lower() in [d.lower() for d in date_columns]:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                # Rename to standard 'date' column
                if col != 'date':
                    df['date'] = df[col]
                break
            except:
                continue
    
    return df

@st.cache_data  
def create_adaptive_features(df):
    """Create features that adapt to any dataset structure"""
    df = df.copy().reset_index(drop=True)
    
    # Ensure we have a date column
    if 'date' not in df.columns:
        # Try to find any date-like column
        date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_candidates:
            df['date'] = pd.to_datetime(df[date_candidates[0]], errors='coerce')
        else:
            # Create dummy dates if no date found
            st.warning("⚠️ No date column found. Using sequential dates.")
            start_date = pd.Timestamp('2024-01-01')
            df['date'] = [start_date + pd.Timedelta(days=i) for i in range(len(df))]
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Basic temporal features
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["quarter"] = df["date"].dt.quarter.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
    df["is_spring_summer"] = df["month"].isin([3, 4, 5, 6, 7, 8]).astype(int)
    
    # Adaptive sustainability detection
    if "material" in df.columns:
        sustainable_keywords = ["organic", "recycled", "hemp", "bamboo", "linen", "tencel", "eco", "green", "sustainable"]
        df["is_sustainable"] = df["material"].str.lower().str.contains(
            '|'.join(sustainable_keywords), na=False
        ).astype(int)
    else:
        df["is_sustainable"] = 0
    
    # Adaptive category detection
    if "category" in df.columns:
        seasonal_keywords = ["shorts", "jacket", "sweater", "coat", "swimwear", "sandals", "boots"]
        df["is_seasonal_category"] = df["category"].str.lower().str.contains(
            '|'.join(seasonal_keywords), na=False
        ).astype(int)
    else:
        df["is_seasonal_category"] = 0
    
    # Create lag features if units/quantity column exists
    quantity_cols = ['units', 'quantity', 'qty', 'sold', 'sales_qty', 'order_quantity']
    quantity_col = None
    
    for col in quantity_cols:
        if col in df.columns:
            quantity_col = col
            break
    
    if quantity_col:
        # Use actual values for lag features
        df = df.sort_values(['date']).reset_index(drop=True)
        base_value = df[quantity_col].mean()
        
        for lag in [1, 2, 4, 8]:
            df[f"units_lag_{lag}"] = df[quantity_col].shift(lag).fillna(base_value)
        
        for window in [4, 8, 12]:
            df[f"units_ma_{window}"] = df[quantity_col].rolling(window=window, min_periods=1).mean()
            df[f"units_std_{window}"] = df[quantity_col].rolling(window=window, min_periods=1).std().fillna(15)
        
        # Standardize column name
        if quantity_col != 'units':
            df['units'] = df[quantity_col]
    else:
        # Create dummy lag features
        base_value = 100
        for lag in [1, 2, 4, 8]:
            df[f"units_lag_{lag}"] = base_value
        for window in [4, 8, 12]:
            df[f"units_ma_{window}"] = base_value
            df[f"units_std_{window}"] = 15
        
        # Create dummy units column if missing
        df['units'] = base_value
    
    return df

def analyze_dataset_structure(df):
    """Analyze and display what was detected in the uploaded dataset"""
    
    st.subheader("📋 Dataset Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        if 'date' in df.columns:
            date_range = df['date'].nunique()
            st.metric("Date Range", f"{date_range} days")
        else:
            st.metric("Date Range", "N/A")
    
    # Show detected columns
    st.write("**🔍 Detected Columns:**")
    
    detected_info = {}
    
    # Check for key columns
    key_columns = ['date', 'sku', 'store', 'units', 'material', 'category', 'price', 'revenue']
    
    for col in key_columns:
        if col in df.columns:
            detected_info[col] = f"✅ {col} (detected)"
        else:
            detected_info[col] = f"❌ {col} (missing)"
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (key, value) in enumerate(detected_info.items()):
            if i < len(detected_info) // 2:
                st.write(value)
    
    with col2:
        for i, (key, value) in enumerate(detected_info.items()):
            if i >= len(detected_info) // 2:
                st.write(value)
    
    # Show additional columns found
    additional_cols = [col for col in df.columns if col not in key_columns and not col.startswith('units_') and not col.startswith('is_')]
    if additional_cols:
        st.write("**📊 Additional Columns Found:**")
        for col in additional_cols:
            st.write(f"• {col}")

def validate_data(df):
    """Comprehensive data validation"""
    errors = []
    warnings = []
    
    # Check for basic data existence
    if len(df) == 0:
        errors.append("Dataset is empty")
        return errors, warnings
    
    # Check for date column (more flexible)
    if 'date' not in df.columns:
        warnings.append("No 'date' column found - will create sequential dates")
    
    # Check for negative values in units
    if "units" in df.columns:
        if (df["units"] < 0).any():
            warnings.append("Negative values found in 'units' column")
        if df["units"].isna().any():
            warnings.append("Missing values found in 'units' column")
    
    # Data range validation
    if len(df) < 10:
        warnings.append("Dataset contains fewer than 10 records - results may be unreliable")
    
    # Check for duplicates
    if df.duplicated().any():
        warnings.append("Duplicate rows found in dataset")
    
    return errors, warnings

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive forecasting metrics"""
    # Handle missing values
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return None
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    
    # Avoid division by zero in MAPE
    non_zero_mask = y_true_clean != 0
    if non_zero_mask.any():
        mape = np.mean(np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R²": r2,
        "Data Points": len(y_true_clean)
    }

def create_guaranteed_charts(df_processed, chart_theme="plotly"):
    """Create charts that WILL display - guaranteed!"""
    charts_created = []
    
    st.write("🔍 **Chart Debug Info:**")
    st.write(f"📊 DataFrame shape: {df_processed.shape}")
    st.write(f"📋 Available columns: {list(df_processed.columns)}")
    
    if len(df_processed) == 0:
        st.error("❌ No data to visualize")
        return []
    
    # 1. ALWAYS show basic overview
    st.subheader("📈 Data Overview")
    
    # Find any numeric column
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        first_numeric = numeric_cols[0]
        st.write(f"📊 Analyzing: **{first_numeric}**")
        
        # Time series if date available
        if 'date' in df_processed.columns:
            try:
                daily_data = df_processed.groupby('date')[first_numeric].sum().reset_index()
                if len(daily_data) > 0:
                    fig1 = px.line(daily_data, x='date', y=first_numeric, 
                                  title=f'{first_numeric.title()} Over Time')
                    fig1.update_layout(template=chart_theme, height=400)
                    st.plotly_chart(fig1, use_container_width=True)
                    charts_created.append("Time Series")
                    st.success("✅ Time series chart created!")
            except Exception as e:
                st.warning(f"Time series error: {e}")
        
        # Histogram of first numeric column
        try:
            fig2 = px.histogram(df_processed, x=first_numeric, 
                              title=f'{first_numeric.title()} Distribution',
                              nbins=30)
            fig2.update_layout(template=chart_theme, height=400)
            st.plotly_chart(fig2, use_container_width=True)
            charts_created.append("Distribution")
            st.success("✅ Distribution chart created!")
        except Exception as e:
            st.warning(f"Histogram error: {e}")
    
    # 2. Material Analysis - FORCE it to show
    if 'material' in df_processed.columns:
        st.subheader("🧵 Material Analysis")
        try:
            # Use any available numeric column
            value_col = 'units' if 'units' in df_processed.columns else numeric_cols[0]
            material_data = df_processed.groupby('material')[value_col].sum().reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig3 = px.pie(material_data, values=value_col, names='material', 
                             title=f'{value_col.title()} by Material')
                fig3.update_layout(template=chart_theme)
                st.plotly_chart(fig3, use_container_width=True)
                st.success("✅ Material pie chart created!")
            
            with col2:
                fig4 = px.bar(material_data, x='material', y=value_col, 
                             title=f'{value_col.title()} by Material Type')
                fig4.update_layout(template=chart_theme)
                st.plotly_chart(fig4, use_container_width=True)
                st.success("✅ Material bar chart created!")
            
            charts_created.append("Material Analysis")
        except Exception as e:
            st.warning(f"Material analysis error: {e}")
    
    # 3. Store Analysis
    if 'store' in df_processed.columns:
        st.subheader("🏪 Store Analysis")
        try:
            value_col = 'units' if 'units' in df_processed.columns else numeric_cols[0]
            store_data = df_processed.groupby('store')[value_col].sum().reset_index()
            
            fig5 = px.bar(store_data, x='store', y=value_col, 
                         title=f'{value_col.title()} by Store')
            fig5.update_xaxes(tickangle=45)
            fig5.update_layout(template=chart_theme)
            st.plotly_chart(fig5, use_container_width=True)
            charts_created.append("Store Analysis")
            st.success("✅ Store analysis chart created!")
        except Exception as e:
            st.warning(f"Store analysis error: {e}")
    
    # 4. Category Analysis
    if 'category' in df_processed.columns:
        st.subheader("🏷️ Category Analysis")
        try:
            value_col = 'units' if 'units' in df_processed.columns else numeric_cols[0]
            category_data = df_processed.groupby('category')[value_col].sum().reset_index()
            
            fig6 = px.bar(category_data, x='category', y=value_col, 
                         title=f'{value_col.title()} by Category')
            fig6.update_xaxes(tickangle=45)
            fig6.update_layout(template=chart_theme)
            st.plotly_chart(fig6, use_container_width=True)
            charts_created.append("Category Analysis")
            st.success("✅ Category analysis chart created!")
        except Exception as e:
            st.warning(f"Category analysis error: {e}")
    
    # 5. Correlation Heatmap - ALWAYS try to show this
    st.subheader("🔬 Data Relationships")
    if len(numeric_cols) >= 2:
        try:
            # Use only first 8 columns to avoid overcrowding
            cols_to_use = numeric_cols[:8]
            corr_data = df_processed[cols_to_use].corr()
            
            fig7 = px.imshow(corr_data, 
                           title='Feature Correlation Heatmap',
                           aspect='auto',
                           color_continuous_scale='RdBu')
            fig7.update_layout(template=chart_theme)
            st.plotly_chart(fig7, use_container_width=True)
            charts_created.append("Correlation")
            st.success("✅ Correlation heatmap created!")
        except Exception as e:
            st.warning(f"Correlation heatmap error: {e}")
    
    # 6. Top SKUs if available
    if 'sku' in df_processed.columns and len(numeric_cols) > 0:
        st.subheader("🏆 Top Performing SKUs")
        try:
            value_col = 'units' if 'units' in df_processed.columns else numeric_cols[0]
            top_skus = df_processed.groupby('sku')[value_col].sum().nlargest(10).reset_index()
            
            fig8 = px.bar(top_skus, x='sku', y=value_col, 
                         title=f'Top 10 SKUs by {value_col.title()}')
            fig8.update_xaxes(tickangle=45)
            fig8.update_layout(template=chart_theme)
            st.plotly_chart(fig8, use_container_width=True)
            charts_created.append("Top SKUs")
            st.success("✅ Top SKUs chart created!")
        except Exception as e:
            st.warning(f"Top SKUs error: {e}")
    
    # 7. Show data table as backup
    st.subheader("📋 Data Sample")
    try:
        st.dataframe(df_processed.head(20), use_container_width=True)
        st.success("✅ Data table displayed!")
    except Exception as e:
        st.error(f"Data table error: {e}")
    
    return charts_created

def create_forecasting_charts(df_processed, y_true, preds, chart_theme="plotly"):
    """Create forecasting-specific charts"""
    
    if 'forecast' not in df_processed.columns or y_true is None:
        return
    
    st.subheader("🎯 Forecasting Analysis")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted scatter
            fig_scatter = px.scatter(df_processed, x='units', y='forecast',
                                   title='Actual vs Predicted Values',
                                   labels={'units': 'Actual', 'forecast': 'Predicted'})
            
            # Add perfect prediction line
            min_val = min(df_processed['units'].min(), df_processed['forecast'].min())
            max_val = max(df_processed['units'].max(), df_processed['forecast'].max())
            fig_scatter.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                                line=dict(color='red', dash='dash'))
            
            fig_scatter.update_layout(template=chart_theme)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Residual analysis
            df_processed['residuals'] = df_processed['units'] - df_processed['forecast']
            
            fig_residuals = px.scatter(df_processed, x='forecast', y='residuals',
                                     title='Residual Analysis',
                                     labels={'forecast': 'Predicted', 'residuals': 'Residuals'})
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            fig_residuals.update_layout(template=chart_theme)
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Error distribution
        fig_error = px.histogram(df_processed, x='residuals',
                               title='Prediction Error Distribution',
                               nbins=30)
        fig_error.update_layout(template=chart_theme)
        st.plotly_chart(fig_error, use_container_width=True)
        
        st.success("✅ Forecasting charts created!")
        
    except Exception as e:
        st.warning(f"Forecasting charts error: {e}")

def calculate_profitability(df_processed):
    """Calculate profitability metrics for products"""
    df_profit = df_processed.copy()
    
    # Calculate profit using multiple methods
    profit_methods = []
    
    # Method 1: If revenue and price columns exist
    if 'revenue' in df_profit.columns and 'price' in df_profit.columns:
        # Assume cost is 60% of price (typical textile margin)
        df_profit['estimated_cost'] = df_profit['price'] * 0.6
        df_profit['profit_per_unit'] = df_profit['price'] - df_profit['estimated_cost']
        df_profit['total_profit_method1'] = df_profit['profit_per_unit'] * df_profit['units']
        df_profit['profit_margin_method1'] = (df_profit['profit_per_unit'] / df_profit['price'] * 100).round(2)
        profit_methods.append("Revenue & Price Analysis")
    
    # Method 2: If only revenue exists
    elif 'revenue' in df_profit.columns:
        # Calculate profit assuming 40% margin
        df_profit['total_profit_method2'] = df_profit['revenue'] * 0.4
        df_profit['profit_per_unit'] = df_profit['total_profit_method2'] / df_profit['units']
        df_profit['profit_margin_method2'] = 40.0  # Assumed margin
        profit_methods.append("Revenue-Based Analysis")
    
    # Method 3: If only price exists
    elif 'price' in df_profit.columns:
        # Estimate cost and calculate profit
        df_profit['estimated_cost'] = df_profit['price'] * 0.6
        df_profit['profit_per_unit'] = df_profit['price'] - df_profit['estimated_cost']
        df_profit['total_profit_method3'] = df_profit['profit_per_unit'] * df_profit['units']
        df_profit['profit_margin_method3'] = (df_profit['profit_per_unit'] / df_profit['price'] * 100).round(2)
        profit_methods.append("Price-Based Analysis")
    
    # Method 4: Estimate based on units and industry averages
    else:
        # Use textile industry averages
        material_profit_rates = {
            'organic_cotton': 0.45, 'recycled_polyester': 0.42, 'hemp': 0.50,
            'bamboo': 0.48, 'linen': 0.46, 'tencel': 0.52,
            'cotton': 0.35, 'polyester': 0.30, 'viscose': 0.32, 'acrylic': 0.25
        }
        
        category_base_prices = {
            't-shirt': 25, 'jeans': 75, 'dress': 85, 'jacket': 120,
            'shorts': 35, 'sweater': 65, 'pants': 55, 'skirt': 45, 'blouse': 50
        }
        
        # Estimate price based on category and material
        df_profit['estimated_price'] = df_profit.apply(lambda row: 
            category_base_prices.get(row.get('category', 't-shirt'), 50) *
            (1.2 if material_profit_rates.get(row.get('material', 'cotton'), 0.35) > 0.4 else 1.0), axis=1)
        
        df_profit['estimated_profit_rate'] = df_profit.apply(lambda row:
            material_profit_rates.get(row.get('material', 'cotton'), 0.35), axis=1)
        
        df_profit['estimated_revenue'] = df_profit['estimated_price'] * df_profit['units']
        df_profit['total_profit_method4'] = df_profit['estimated_revenue'] * df_profit['estimated_profit_rate']
        df_profit['profit_per_unit'] = df_profit['total_profit_method4'] / df_profit['units']
        df_profit['profit_margin_method4'] = (df_profit['estimated_profit_rate'] * 100).round(2)
        profit_methods.append("Industry Benchmark Analysis")
    
    return df_profit, profit_methods

def create_profitability_charts(df_profit, profit_methods, chart_theme="plotly"):
    """Create comprehensive profitability analysis charts"""
    
    st.header("💰 Profitability Analysis")
    st.write(f"**Analysis Methods Used:** {', '.join(profit_methods)}")
    
    # Determine which profit column to use
    profit_col = None
    margin_col = None
    
    if 'total_profit_method1' in df_profit.columns:
        profit_col = 'total_profit_method1'
        margin_col = 'profit_margin_method1'
    elif 'total_profit_method2' in df_profit.columns:
        profit_col = 'total_profit_method2'
        margin_col = 'profit_margin_method2'
    elif 'total_profit_method3' in df_profit.columns:
        profit_col = 'total_profit_method3'
        margin_col = 'profit_margin_method3'
    elif 'total_profit_method4' in df_profit.columns:
        profit_col = 'total_profit_method4'
        margin_col = 'profit_margin_method4'
    
    if profit_col is None:
        st.error("❌ Could not calculate profitability metrics")
        return df_profit
    
    # 1. Top Profitable Products by SKU
    st.subheader("🏆 Most Profitable Products")
    
    if 'sku' in df_profit.columns:
        sku_profits = df_profit.groupby('sku').agg({
            profit_col: 'sum',
            'profit_per_unit': 'mean',
            'units': 'sum'
        }).round(2).reset_index()
        
        sku_profits.columns = ['SKU', 'Total_Profit', 'Avg_Profit_Per_Unit', 'Total_Units']
        sku_profits = sku_profits.sort_values('Total_Profit', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 profitable SKUs
            top_skus = sku_profits.head(10)
            fig1 = px.bar(top_skus, x='SKU', y='Total_Profit',
                         title='Top 10 Most Profitable SKUs',
                         text='Total_Profit')
            fig1.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig1.update_xaxes(tickangle=45)
            fig1.update_layout(template=chart_theme)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Show top performers table
            st.write("**🥇 Top Profitable SKUs:**")
            top_skus_display = top_skus.copy()
            top_skus_display['Total_Profit'] = top_skus_display['Total_Profit'].apply(lambda x: f"${x:,.2f}")
            top_skus_display['Avg_Profit_Per_Unit'] = top_skus_display['Avg_Profit_Per_Unit'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(top_skus_display, use_container_width=True)
        
        with col2:
            # Bottom 10 profitable SKUs
            bottom_skus = sku_profits.tail(10)
            fig2 = px.bar(bottom_skus, x='SKU', y='Total_Profit',
                         title='Bottom 10 Least Profitable SKUs',
                         text='Total_Profit',
                         color_discrete_sequence=['red'])
            fig2.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig2.update_xaxes(tickangle=45)
            fig2.update_layout(template=chart_theme)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Show bottom performers table
            st.write("**📉 Least Profitable SKUs:**")
            bottom_skus_display = bottom_skus.copy()
            bottom_skus_display['Total_Profit'] = bottom_skus_display['Total_Profit'].apply(lambda x: f"${x:,.2f}")
            bottom_skus_display['Avg_Profit_Per_Unit'] = bottom_skus_display['Avg_Profit_Per_Unit'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(bottom_skus_display, use_container_width=True)
    
    # 2. Profitability by Material
    if 'material' in df_profit.columns:
        st.subheader("🧵 Profitability by Material Type")
        
        material_profits = df_profit.groupby('material').agg({
            profit_col: 'sum',
            'profit_per_unit': 'mean',
            'units': 'sum'
        }).round(2).reset_index()
        
        material_profits.columns = ['Material', 'Total_Profit', 'Avg_Profit_Per_Unit', 'Total_Units']
        material_profits = material_profits.sort_values('Total_Profit', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Material profit pie chart
            fig3 = px.pie(material_profits, values='Total_Profit', names='Material',
                         title='Profit Distribution by Material')
            fig3.update_layout(template=chart_theme)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Material profit per unit
            fig4 = px.bar(material_profits, x='Material', y='Avg_Profit_Per_Unit',
                         title='Average Profit Per Unit by Material',
                         text='Avg_Profit_Per_Unit')
            fig4.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig4.update_xaxes(tickangle=45)
            fig4.update_layout(template=chart_theme)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Material profitability table
        st.write("**📊 Material Profitability Summary:**")
        material_display = material_profits.copy()
        material_display['Total_Profit'] = material_display['Total_Profit'].apply(lambda x: f"${x:,.2f}")
        material_display['Avg_Profit_Per_Unit'] = material_display['Avg_Profit_Per_Unit'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(material_display, use_container_width=True)
    
    # 3. Profitability by Category
    if 'category' in df_profit.columns:
        st.subheader("🏷️ Profitability by Product Category")
        
        category_profits = df_profit.groupby('category').agg({
            profit_col: 'sum',
            'profit_per_unit': 'mean',
            'units': 'sum'
        }).round(2).reset_index()
        
        category_profits.columns = ['Category', 'Total_Profit', 'Avg_Profit_Per_Unit', 'Total_Units']
        category_profits = category_profits.sort_values('Total_Profit', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category profit bar chart
            fig5 = px.bar(category_profits, x='Category', y='Total_Profit',
                         title='Total Profit by Product Category',
                         text='Total_Profit')
            fig5.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig5.update_xaxes(tickangle=45)
            fig5.update_layout(template=chart_theme)
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # Category profit margin scatter
            if margin_col in df_profit.columns:
                category_margin = df_profit.groupby('category')[margin_col].mean().reset_index()
                category_margin.columns = ['Category', 'Avg_Profit_Margin']
                
                fig6 = px.scatter(category_profits.merge(category_margin), 
                                 x='Total_Units', y='Total_Profit', 
                                 size='Avg_Profit_Per_Unit',
                                 hover_name='Category',
                                 title='Profit vs Volume by Category')
                fig6.update_layout(template=chart_theme)
                st.plotly_chart(fig6, use_container_width=True)
    
    # 4. Profitability by Store
    if 'store' in df_profit.columns:
        st.subheader("🏪 Store Profitability Analysis")
        
        store_profits = df_profit.groupby('store').agg({
            profit_col: 'sum',
            'profit_per_unit': 'mean',
            'units': 'sum'
        }).round(2).reset_index()
        
        store_profits.columns = ['Store', 'Total_Profit', 'Avg_Profit_Per_Unit', 'Total_Units']
        store_profits = store_profits.sort_values('Total_Profit', ascending=False)
        
        # Store profit comparison
        fig7 = px.bar(store_profits, x='Store', y='Total_Profit',
                     title='Total Profit by Store',
                     text='Total_Profit')
        fig7.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig7.update_xaxes(tickangle=45)
        fig7.update_layout(template=chart_theme)
        st.plotly_chart(fig7, use_container_width=True)
        
        # Store performance table
        st.write("**🏆 Store Performance Ranking:**")
        store_display = store_profits.copy()
        store_display['Total_Profit'] = store_display['Total_Profit'].apply(lambda x: f"${x:,.2f}")
        store_display['Avg_Profit_Per_Unit'] = store_display['Avg_Profit_Per_Unit'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(store_display, use_container_width=True)
    
    # 5. Profit Trends Over Time
    if 'date' in df_profit.columns:
        st.subheader("📈 Profit Trends Over Time")
        
        daily_profits = df_profit.groupby('date')[profit_col].sum().reset_index()
        daily_profits.columns = ['Date', 'Daily_Profit']
        
        fig8 = px.line(daily_profits, x='Date', y='Daily_Profit',
                      title='Daily Profit Trends')
        fig8.update_layout(template=chart_theme)
        st.plotly_chart(fig8, use_container_width=True)
        
        # Monthly profit summary
        df_profit['month_year'] = df_profit['date'].dt.to_period('M').astype(str)
        monthly_profits = df_profit.groupby('month_year')[profit_col].sum().reset_index()
        monthly_profits.columns = ['Month', 'Monthly_Profit']
        
        fig9 = px.bar(monthly_profits, x='Month', y='Monthly_Profit',
                     title='Monthly Profit Summary')
        fig9.update_xaxes(tickangle=45)
        fig9.update_layout(template=chart_theme)
        st.plotly_chart(fig9, use_container_width=True)
    
    # 6. Key Insights Summary
    st.subheader("💡 Key Profitability Insights")
    
    total_profit = df_profit[profit_col].sum()
    avg_profit_per_unit = df_profit['profit_per_unit'].mean()
    total_units = df_profit['units'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Profit", f"${total_profit:,.2f}")
    with col2:
        st.metric("Avg Profit/Unit", f"${avg_profit_per_unit:.2f}")
    with col3:
        st.metric("Total Units Sold", f"{total_units:,}")
    with col4:
        st.metric("Overall Margin", f"{(total_profit/(total_profit/0.4) if total_profit > 0 else 0)*100:.1f}%")
    
    # Insights bullets
    insights = []
    
    if 'sku' in df_profit.columns:
        most_profitable_sku = sku_profits.iloc[0]['SKU']
        most_profitable_amount = sku_profits.iloc[0]['Total_Profit']
        insights.append(f"🥇 **Most Profitable SKU:** {most_profitable_sku} (${most_profitable_amount:,.2f})")
    
    if 'material' in df_profit.columns:
        most_profitable_material = material_profits.iloc[0]['Material']
        insights.append(f"🧵 **Most Profitable Material:** {most_profitable_material}")
    
    if 'category' in df_profit.columns:
        most_profitable_category = category_profits.iloc[0]['Category']
        insights.append(f"🏷️ **Most Profitable Category:** {most_profitable_category}")
    
    if 'store' in df_profit.columns:
        most_profitable_store = store_profits.iloc[0]['Store']
        insights.append(f"🏪 **Top Performing Store:** {most_profitable_store}")
    
    for insight in insights:
        st.write(insight)
    
    return df_profit

# Main app
def main():
    st.title("🌱 Sustainable Textile Demand Forecasting")
    st.markdown("Advanced ML-powered forecasting with comprehensive profitability analysis - **Universal Format Support**")
    
    # Load model
    model, model_error = load_model()
    
    if model_error:
        st.error(f"❌ {model_error}")
        st.info("💡 Charts and data analysis are still available without the model!")
        model = None
    else:
        st.success("✅ Model loaded successfully")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Model information
        st.subheader("Model Info")
        if model:
            st.info(f"**Model Type:** {type(model).__name__}")
        else:
            st.warning("**Model:** Not available")
        
        # Analysis options
        st.subheader("Analysis Options")
        show_feature_importance = st.checkbox("Show Feature Importance", value=True)
        show_sustainability_analysis = st.checkbox("Show Sustainability Analysis", value=True)
        show_profitability_analysis = st.checkbox("Show Profitability Analysis", value=True)
        show_advanced_charts = st.checkbox("Show Advanced Visualizations", value=True)
        
        # Chart options
        st.subheader("📊 Chart Options")
        chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2"])
        show_debug_info = st.checkbox("Show Debug Info", value=False)
    
    # File upload section
    st.header("📁 Data Upload - Universal Format Support")
    uploaded_file = st.file_uploader(
        "Upload your textile dataset", 
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Supports CSV, Excel, JSON, and Parquet formats. Automatic profitability analysis included!"
    )
    
    if uploaded_file is not None:
        try:
            # Show file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            with st.expander("📄 File Information"):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            
            # Process file
            with st.spinner("🔄 Processing uploaded file..."):
                df = detect_and_process_file(uploaded_file)
            
            if df is None:
                st.stop()
            
            st.success("✅ File processed successfully!")
            
            # IMMEDIATE CHART DISPLAY TEST
            st.header("🔍 Quick Data Visualization Test")
            
            # Test chart with your data
            if len(df) > 0:
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_columns) > 0:
                    first_numeric = numeric_columns[0]
                    st.write(f"📊 Testing chart with column: **{first_numeric}**")
                    
                    try:
                        if 'date' in df.columns:
                            # Time series test
                            daily_test = df.groupby('date')[first_numeric].sum().reset_index()
                            fig_test = px.line(daily_test, x='date', y=first_numeric, 
                                             title=f'TEST: {first_numeric} Over Time')
                            fig_test.update_layout(height=400)
                            st.plotly_chart(fig_test, use_container_width=True)
                            st.success("✅ TEST CHART DISPLAYED SUCCESSFULLY!")
                        else:
                            # Histogram test
                            fig_test = px.histogram(df, x=first_numeric, 
                                                  title=f'TEST: {first_numeric} Distribution')
                            fig_test.update_layout(height=400)
                            st.plotly_chart(fig_test, use_container_width=True)
                            st.success("✅ TEST CHART DISPLAYED SUCCESSFULLY!")
                    except Exception as e:
                        st.error(f"❌ Test chart error: {e}")
                        if show_debug_info:
                            st.write("Raw data info:")
                            st.write(f"Shape: {df.shape}")
                            st.write(f"Columns: {list(df.columns)}")
                            st.write(f"First few rows:")
                            st.dataframe(df.head())
            
            # Analyze dataset structure
            analyze_dataset_structure(df)
            
            # Data validation
            errors, warnings = validate_data(df)
            
            if errors:
                st.error("❌ Data validation failed:")
                for error in errors:
                    st.write(f"• {error}")
                st.stop()
            
            if warnings:
                st.warning("⚠️ Data quality warnings:")
                for warning in warnings:
                    st.write(f"• {warning}")
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Feature engineering
            with st.spinner("🔧 Creating adaptive features..."):
                df_processed = create_adaptive_features(df)
            
            st.success("✅ Adaptive features created successfully!")
            
            # GUARANTEED CHARTS DISPLAY
            st.header("📊 Comprehensive Data Analysis & Visualizations")
            
            # Create charts that WILL display
            charts_created = create_guaranteed_charts(df_processed, chart_theme)
            
            if len(charts_created) > 0:
                st.info(f"📈 Successfully generated {len(charts_created)} chart types: {', '.join(charts_created)}")
            else:
                st.warning("⚠️ No charts could be generated")
            
            # PROFITABILITY ANALYSIS - Add this section
            if show_profitability_analysis:
                try:
                    # Calculate profitability
                    df_profit, profit_methods = calculate_profitability(df_processed)
                    
                    if len(profit_methods) > 0:
                        # Create profitability charts
                        df_profit = create_profitability_charts(df_profit, profit_methods, chart_theme)
                        
                        st.success("✅ Profitability analysis completed!")
                        
                    else:
                        st.warning("⚠️ Could not perform profitability analysis - insufficient data")
                        st.info("💡 For profitability analysis, please include columns like: price, revenue, or cost data")

                except Exception as e:
                    st.error(f"❌ Profitability analysis error: {str(e)}")
                    st.info("💡 This feature works best with price, revenue, or cost data in your dataset")
            
            # Try predictions if model available
            if model is not None:
                # Define features for prediction
                excluded_cols = ["date", "sku", "store", "units", "material", "category", "price", "revenue"]
                FEATURES = [c for c in df_processed.columns if c not in excluded_cols]
                
                if len(FEATURES) > 0:
                    # Handle missing values in features
                    X = df_processed[FEATURES].fillna(0)
                    y_true = df_processed["units"] if "units" in df_processed else None
                    
                    # Make predictions
                    try:
                        with st.spinner("🔮 Generating predictions..."):
                            preds = model.predict(X)
                            df_processed["forecast"] = preds
                            st.success("✅ Predictions generated successfully")
                            
                            # Results section
                            st.header("📈 Forecast Results")
                            
                            # Display results
                            result_columns = ["date", "sku", "store", "units", "forecast"]
                            if "material" in df_processed:
                                result_columns.append("material")
                            
                            available_columns = [col for col in result_columns if col in df_processed.columns]
                            st.dataframe(df_processed[available_columns].head(20), use_container_width=True)
                            
                            # Performance metrics
                            if y_true is not None:
                                st.subheader("🎯 Model Performance")
                                
                                metrics = calculate_metrics(y_true, preds)
                                
                                if metrics:
                                    col1, col2, col3, col4, col5 = st.columns(5)
                                    
                                    with col1:
                                        st.metric("MAE", f"{metrics['MAE']:.2f}")
                                    with col2:
                                        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                                    with col3:
                                        if metrics['MAPE'] != float('inf'):
                                            st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                                        else:
                                            st.metric("MAPE", "N/A")
                                    with col4:
                                        st.metric("R²", f"{metrics['R²']:.3f}")
                                    with col5:
                                        if metrics['MAPE'] != float('inf'):
                                            accuracy = max(0, 100 - metrics['MAPE'])
                                            st.metric("Accuracy", f"{accuracy:.1f}%")
                                        else:
                                            st.metric("Accuracy", "N/A")
                            
                            # Forecasting charts
                            create_forecasting_charts(df_processed, y_true, preds, chart_theme)
                            
                    except Exception as e:
                        st.warning(f"⚠️ Prediction failed: {str(e)}")
                        st.info("📊 Data analysis charts are still available above")
                else:
                    st.warning("⚠️ No suitable features found for prediction")
            
            # Export functionality
            st.subheader("📤 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV export
                export_cols = ["date", "sku", "store", "units"]
                if "forecast" in df_processed.columns:
                    export_cols.append("forecast")
                if "material" in df_processed.columns:
                    export_cols.append("material")
                
                available_export_cols = [col for col in export_cols if col in df_processed.columns]
                csv_data = df_processed[available_export_cols].to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Analysis CSV",
                    data=csv_data,
                    file_name=f"textile_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Profitability export
                if show_profitability_analysis and 'df_profit' in locals():
                    profit_export_cols = ['date', 'sku', 'store', 'units']
                    if 'material' in df_profit.columns:
                        profit_export_cols.append('material')
                    if 'category' in df_profit.columns:
                        profit_export_cols.append('category')
                    if 'price' in df_profit.columns:
                        profit_export_cols.append('price')
                    if 'revenue' in df_profit.columns:
                        profit_export_cols.append('revenue')
                    
                    profit_export_cols.extend(['profit_per_unit'])
                    
                    # Add the calculated profit column
                    if 'total_profit_method1' in df_profit.columns:
                        profit_export_cols.append('total_profit_method1')
                    elif 'total_profit_method2' in df_profit.columns:
                        profit_export_cols.append('total_profit_method2')
                    elif 'total_profit_method3' in df_profit.columns:
                        profit_export_cols.append('total_profit_method3')
                    elif 'total_profit_method4' in df_profit.columns:
                        profit_export_cols.append('total_profit_method4')
                    
                    available_profit_cols = [col for col in profit_export_cols if col in df_profit.columns]
                    profit_csv = df_profit[available_profit_cols].to_csv(index=False)
                    
                    st.download_button(
                        label="💰 Download Profitability Analysis",
                        data=profit_csv,
                        file_name=f"profitability_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                # Summary export
                summary_data = {
                    "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_records": len(df_processed),
                    "date_range": f"{df_processed['date'].min()} to {df_processed['date'].max()}" if 'date' in df_processed.columns else "N/A",
                    "charts_generated": len(charts_created),
                    "chart_types": ', '.join(charts_created),
                    "profitability_analysis": "Yes" if show_profitability_analysis else "No"
                }
                
                summary_str = "\n".join([f"{k}: {v}" for k, v in summary_data.items()])
                st.download_button(
                    label="📋 Download Summary",
                    data=summary_str,
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            if show_debug_info:
                st.write("**Debug Info:**")
                st.write(f"Error type: {type(e).__name__}")
                import traceback
                st.text(traceback.format_exc())
    
    else:
        st.info("👆 Please upload your textile dataset to begin comprehensive analysis")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'sku': ['SKU001', 'SKU001', 'SKU002'],
            'store': ['Store_A', 'Store_A', 'Store_B'],
            'units': [150, 200, 75],
            'material': ['organic_cotton', 'polyester', 'hemp'],
            'category': ['t-shirt', 't-shirt', 'jeans'],
            'price': [25.00, 30.00, 75.00],
            'revenue': [3750.00, 6000.00, 5625.00]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        st.caption("💡 Include price/revenue columns for automatic profitability analysis!")
        
        # Show features
        st.subheader("🚀 App Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Data Analysis:**")
            st.write("• Universal file format support")
            st.write("• Automatic chart generation")
            st.write("• Material & category analysis")
            st.write("• Store performance tracking")
            st.write("• Time series visualization")
        
        with col2:
            st.write("**💰 Profitability Analysis:**")
            st.write("• Most/least profitable products")
            st.write("• Material profitability ranking")
            st.write("• Category profit analysis")
            st.write("• Store profitability comparison")
            st.write("• Profit trend visualization")

if __name__ == "__main__":
    main()
