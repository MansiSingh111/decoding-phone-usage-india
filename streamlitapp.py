import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import joblib
import pickle 

st.set_page_config(page_title="The Analysis - Phone Usage in India", layout='wide')

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Introduction",
        "EDA & Visualisation", 
        "Classification Prediction",
        "Clustering Analysis"
    ]
)

# load data

def load_data():
    return pd.read_csv(r"C:\Users\Mansi\.vscode\Phone_Usage_Pattern\cleaned_phone_usage_india.csv")
df = load_data()

# PAGE 1
if page == "Introduction":
    st.title(" The Analysis - Phone Usage in India")
    st.markdown("""
**Welcome to the interactive dashboard!**
Analyzing phone usage patterns in India with advanced machine learning:
                
- **Classification:** Predict unknown usage records after training
- **Clustering:** Discover how features affect primary phone usage
- **EDA:** Built-in analysis tools for easy data understanding
                
Navigate pages with the sidebar!
""")
    st.header("Project Overview")
    st.markdown("""
1. Data Cleaning & Encoding
2. Exploratory Data Analysis (EDA)
3. Classification (with 5 trained models)
4. Unsupervised Clustering 
""")
    st.info("Dataset includes: demographics, device type, usage time, app preferences, and more.")


# Page 2:
elif page == "EDA & Visualisation":
    st.title("EDA & Visualisation")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Feature Distribution
    st.subheader("Feature Distribution")
    feature = st.selectbox("Choose a feature", df.select_dtypes(include=[np.number, "object"]).columns)
    if df[feature].dtype in [np.int64, np.float64]:
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax )
        st.pyplot(fig)
    else:
        st.bar_chart(df[feature].value_counts())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    age_means = df.groupby('Age')['Screen Time (hrs/day)'].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(age_means.index, age_means.values, marker='o')
    ax.set_xlabel('Age')
    ax.set_ylabel('Average Screen Time (hrs/day)')
    ax.set_title('Average Screen Time by Age')
    ax.grid(True)
    st.pyplot(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x='Screen Time (hrs/day)', hue='Primary Use', multiple='stack', bins=20, ax=ax)
    ax.set_title('Histogram of Screen Time by Primary Use')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Line Graph
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_data_usage = df.groupby('Primary Use')['Data Usage (GB/month)'].mean().sort_values(ascending=False)
    avg_data_usage.plot(kind='line', marker='o', ax=ax)
    ax.set_xlabel('Primary Use')
    ax.set_ylabel('Average Data Usage (GB/month)')
    ax.set_title('Average Data Usage by Primary Use')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Box plot
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x='Primary Use', y='Number of Apps Installed')
    ax.set_title('Number of Apps installed by Primary Use')
    plt.xticks(rotation=30)
    st.pyplot(fig)


# Page 3:
elif page == "Classification Prediction":
    st.title("⚡ Classification Model Prediction (Manual Entry)")
    st.write("Enter your features to predict your primary use category.")

    # Define which features are numeric and which are categorical
    numerical_features = [
       "Age", "Number of Apps Installed", "Screen Time (hrs/day)", "Data Usage (GB/month)", "Calls Duration (mins/day)", "Social Media Time (hrs/day)", "E-commerce Spend (INR/month)", "Streaming Time (hrs/day)", "Gaming Time (hrs/day)", "Monthly Recharge Cost (INR)"
    ]
    categorical_features = [
     "Gender", "Location", "OS", "Phone Brand"
    ]

    integer_features = ["Age", "Number of Apps Installed"]


    feature_options = {
    "Phone Brand": ["Nokia", "Oneplus", "Xiaomi", "Vivo", "Apple", "Samsung", "Realme", "Google Pixel", "Motorola", "Oppo"],
    "Gender": ["Male", "Female", "Other"],
    "Location": ["Jaipur", "Pune", "Chennai", "Kolkata", "Bangalore", "Ahmedabad", "Delhi", "Mumbai", "Lucknow", "Hyderabad"],
    "OS": ["Android", "iOS"]
     }

    # Mapping from predicted numbers to actual usage names
    usage_map = {
    0: "Education",
    1: "Entertainment",
    2: "Gaming",
    3: "Social media",
    4: "Work"
      }

    # Load the dictionary of pipelines
    with open(r"C:\Users\Mansi\.vscode\Phone_Usage_Pattern\classifier_pipelines.pkl", 'rb') as f:
        pipelines = pickle.load(f)

    model = pipelines['Logistic Regression']

    # Collect user input
    st.subheader("Enter Phone Usage Features")
    user_input = {}

    for feat in numerical_features:
        if feat in integer_features:
            # Force integer input from 0 to 50
            user_input[feat] = st.number_input(
                feat, min_value=0, max_value=50, value=25, step=1
        )
        else:
            # For continuous numeric input, use slider
            user_input[feat] = st.slider(
                feat, min_value=0.0, max_value=100.0, value=10.0
            )
    for feat in categorical_features:
        options = feature_options.get(feat, [])
        user_input[feat] = st.selectbox(feat, options)

    if st.button("Predict Primary Usage"):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        usage_name = usage_map.get(pred, f"Type {pred}")
        st.success(f"Predicted Primary Usage: {usage_name}")


elif page == "Clustering Analysis":
    st.title("🔎 Clustering Assignment")
    st.write("Enter your phone usage features to find your cluster group.")
    
    numerical_features = [
        "Age", "Number of Apps Installed", "Screen Time (hrs/day)", "Data Usage (GB/month)",
        "Calls Duration (mins/day)", "Social Media Time (hrs/day)", "E-commerce Spend (INR/month)",
        "Streaming Time (hrs/day)", "Gaming Time (hrs/day)", "Monthly Recharge Cost (INR)"
    ]
    categorical_features = ["Location", "Phone Brand", "Gender", "OS", "Primary Use"]  # in order for processing
    integer_features = ["Age", "Number of Apps Installed"]
    feature_options = {
        "Phone Brand": ["Nokia", "Oneplus", "Xiaomi", "Vivo", "Apple", "Samsung", "Realme", "Google Pixel", "Motorola", "Oppo"],
        "Gender": ["Male", "Female", "Other"],
        "Location": ["Jaipur", "Pune", "Chennai", "Kolkata", "Bangalore", "Ahmedabad", "Delhi", "Mumbai", "Lucknow", "Hyderabad"],
        "OS": ["Android", "iOS"],
        "Primary Use": ["Entertainment", "Gaming", "Work", "Education", "Social Media"]
    }

    # --- Prepare Label Encoders for Location, Phone Brand ---
    location_map = {v: i for i, v in enumerate(feature_options["Location"])}
    phonebrand_map = {v: i for i, v in enumerate(feature_options["Phone Brand"])}
    primaryuse_map = {v: i for i, v in enumerate(feature_options["Primary Use"])}

    # --- Cluster descriptions ---
    cluster_desc = {
        0: "Heavy Social Media Users",
        1: "Moderate Multi-Activity Users",
        2: "Entertainment & Gaming Focused",
        3: "Low Usage / High Spend"
    }

    # --- Load model ---
    import pickle
    with open(r"C:\Users\Mansi\.vscode\Phone_Usage_Pattern\best_kmeans_model.pkl", 'rb') as f:
        kmeans_model = pickle.load(f)

    st.subheader("Enter Phone Usage Features")
    user_input = {}
    for feat in numerical_features:
        if feat in integer_features:
            user_input[feat] = st.number_input(feat, min_value=0, max_value=50, value=25, step=1)
        else:
            user_input[feat] = st.slider(feat, min_value=0.0, max_value=100.0, value=10.0)
    for feat in categorical_features:
        options = feature_options.get(feat, [])
        user_input[feat] = st.selectbox(feat, options)

    # --- Feature engineering for prediction ---    
    if st.button("Find My Cluster"):
        # Start with numericals, as-is
        encoded_row = []

        for feat in numerical_features:
            encoded_row.append(user_input[feat])

        # Label encode Location, Phone Brand
        encoded_row.append(location_map[user_input["Location"]])
        encoded_row.append(phonebrand_map[user_input["Phone Brand"]])
        encoded_row.append(primaryuse_map[user_input["Primary Use"]])

        # One-hot encode Gender, OS (drop original col)
        for cat in feature_options["Gender"]:
            encoded_row.append(1 if user_input["Gender"] == cat else 0)
        for cat in feature_options["OS"]:
            encoded_row.append(1 if user_input["OS"] == cat else 0)

        # Convert list to DataFrame with correct shape
        input_df = pd.DataFrame([encoded_row])

        # Predict and display
        cluster_num = kmeans_model.predict(input_df)[0]
        desc = cluster_desc.get(cluster_num, "No description available")
        st.success(f"You belong to Cluster: {cluster_num} - {desc}")
