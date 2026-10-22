import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

st.set_page_config(page_title="Traffic Congestion and Road Safety Risk Intelligence", layout="wide")
st.title(" Traffic Congestion and Road Safety Risk Intelligence")
st.write("Analyze traffic data, predict congestion, compare machine learning models, and find traffic patterns.")

# Load the CSV stored beside this Python file.
df = pd.read_csv("Banglore_traffic_Dataset.csv")
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Month"] = df["Date"].dt.month
# Monday = 0 and Sunday = 6
df["Day of Week"] = df["Date"].dt.dayofweek

# Create a simple incident-count category for classification.
df["Risk Category"] = np.where(df["Incident Reports"] >= 2, "Higher incident count", "Lower incident count")

# These columns are used to predict the numerical congestion level.
regression_columns = [
    "Area Name", "Road/Intersection Name", "Traffic Volume", "Average Speed",
    "Travel Time Index", "Road Capacity Utilization", "Environmental Impact",
    "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage",
    "Pedestrian and Cyclist Count", "Weather Conditions",
    "Roadwork and Construction Activity", "Month", "Day of Week"
]

# Incident Reports is not an input to the classifier because it is used to create Risk Category.
classification_columns = regression_columns + ["Congestion Level"]

# Prepare the data for training.
regression_data = df.dropna(subset=["Congestion Level"]).copy()
X_reg = regression_data[regression_columns]
y_reg = regression_data["Congestion Level"]

classification_data = df.dropna(subset=["Risk Category"]).copy()
X_class = classification_data[classification_columns]
y_class = classification_data["Risk Category"]

numeric_reg = X_reg.select_dtypes(include=np.number).columns.tolist()
categorical_reg = X_reg.select_dtypes(exclude=np.number).columns.tolist()
numeric_class = X_class.select_dtypes(include=np.number).columns.tolist()
categorical_class = X_class.select_dtypes(exclude=np.number).columns.tolist()

# Fill missing numeric values, scale numbers, and encode text categories.
regression_preparation = ColumnTransformer([
    ("numbers", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_reg),
    ("categories", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_reg)
])
classification_preparation = ColumnTransformer([
    ("numbers", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_class),
    ("categories", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical_class)
])

# Train simple linear regression using only average speed.
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
simple_model = Pipeline([("fill", SimpleImputer(strategy="median")), ("model", LinearRegression())])
simple_model.fit(X_train_reg[["Average Speed"]], y_train_reg)
simple_predictions = simple_model.predict(X_test_reg[["Average Speed"]])

# Train multiple linear regression using all listed regression features.
multiple_model = Pipeline([("prepare", regression_preparation), ("model", LinearRegression())])
multiple_model.fit(X_train_reg, y_train_reg)
multiple_predictions = multiple_model.predict(X_test_reg)

# Split classification data while keeping class proportions similar.
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Train all required classifiers.
classifier_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8, random_state=42), n_estimators=50, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
}

trained_classifiers = {}
classifier_scores = []
for name, model in classifier_models.items():
    pipeline = Pipeline([("prepare", classification_preparation), ("model", model)])
    pipeline.fit(X_train_class, y_train_class)
    trained_classifiers[name] = pipeline
    predictions = pipeline.predict(X_test_class)
    classifier_scores.append({"Model": name, "Accuracy": accuracy_score(y_test_class, predictions)})


tab1, tab2, tab3, tab4, tab5 = st.tabs([" EDA Dashboard", " Predictions", " Model Evaluation", " Clustering", " Dataset"])

with tab1:
    st.subheader("Traffic data overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Areas", df["Area Name"].nunique())
    c3.metric("Roads / intersections", df["Road/Intersection Name"].nunique())
    c4.metric("Average congestion", f"{df['Congestion Level'].mean():.2f}")

    chart_type = st.selectbox("Choose a chart", ["Congestion distribution", "Average congestion by area", "Traffic volume vs average speed", "Correlation heatmap", "Incident reports", "Weather conditions"])
    if chart_type == "Congestion distribution":
        fig, ax = plt.subplots()
        ax.hist(df["Congestion Level"].dropna(), bins=25, edgecolor="black")
        ax.set_xlabel("Congestion Level")
        ax.set_ylabel("Number of records")
        ax.set_title("Congestion distribution")
        st.pyplot(fig)
    elif chart_type == "Average congestion by area":
        fig, ax = plt.subplots()
        df.groupby("Area Name")["Congestion Level"].mean().sort_values().plot(kind="bar", ax=ax)
        ax.set_ylabel("Average congestion")
        ax.set_title("Average congestion by area")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
    elif chart_type == "Traffic volume vs average speed":
        fig, ax = plt.subplots()
        ax.scatter(df["Traffic Volume"], df["Average Speed"], alpha=0.4)
        ax.set_xlabel("Traffic Volume")
        ax.set_ylabel("Average Speed")
        ax.set_title("Traffic volume and speed")
        st.pyplot(fig)
    elif chart_type == "Correlation heatmap":
        corr = df.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlation heatmap")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
    elif chart_type == "Incident reports":
        fig, ax = plt.subplots()
        df["Incident Reports"].value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_xlabel("Incident reports")
        ax.set_ylabel("Number of records")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        df["Weather Conditions"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Weather")
        ax.set_ylabel("Number of records")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Missing values and data types")
    st.dataframe(pd.DataFrame({"Missing values": df.isnull().sum(), "Data type": df.dtypes.astype(str)}), use_container_width=True)

    st.subheader("Original categorical data")
    category_columns = ["Area Name", "Weather Conditions", "Roadwork and Construction Activity"]
    st.dataframe(df[category_columns].head(8), use_container_width=True)

    st.subheader("Data after One-Hot Encoding")
    encoded_data = pd.get_dummies(df[category_columns].head(8), columns=category_columns, dtype=int)
    st.write("Each category becomes its own column. 1 means yes and 0 means no.")
    st.dataframe(encoded_data, use_container_width=True)
    st.download_button("Download encoded preview", encoded_data.to_csv(index=False), "encoded_traffic_data.csv", "text/csv")

with tab2:
    st.subheader("Predict traffic using all inputs")
    st.write("Enter the traffic conditions. The trained models will estimate congestion and classify the incident-count category.")

    left, right = st.columns(2)
    with left:
        area = st.selectbox("Area Name", sorted(df["Area Name"].dropna().unique()))
        road = st.selectbox("Road / Intersection Name", sorted(df["Road/Intersection Name"].dropna().unique()))
        traffic_volume = st.number_input("Traffic Volume", min_value=0, value=int(df["Traffic Volume"].median()), step=100)
        average_speed = st.number_input("Average Speed", min_value=0.0, value=float(df["Average Speed"].median()), step=1.0)
        travel_time_index = st.number_input("Travel Time Index", min_value=0.0, value=float(df["Travel Time Index"].median()), step=0.1)
        road_capacity = st.number_input("Road Capacity Utilization", min_value=0.0, value=float(df["Road Capacity Utilization"].median()), step=1.0)
        environmental_impact = st.number_input("Environmental Impact", min_value=0.0, value=float(df["Environmental Impact"].median()), step=1.0)
        public_transport = st.number_input("Public Transport Usage", min_value=0.0, value=float(df["Public Transport Usage"].median()), step=1.0)
    with right:
        signal_compliance = st.number_input("Traffic Signal Compliance", min_value=0.0, value=float(df["Traffic Signal Compliance"].median()), step=1.0)
        parking_usage = st.number_input("Parking Usage", min_value=0.0, value=float(df["Parking Usage"].median()), step=1.0)
        pedestrian_count = st.number_input("Pedestrian and Cyclist Count", min_value=0, value=int(df["Pedestrian and Cyclist Count"].median()), step=1)
        weather = st.selectbox("Weather Conditions", sorted(df["Weather Conditions"].dropna().unique()))
        roadwork = st.selectbox("Roadwork and Construction Activity", sorted(df["Roadwork and Construction Activity"].dropna().unique()))
        month = st.selectbox("Month", list(range(1, 13)), index=5)
        day = st.selectbox("Day of Week (0=Monday, 6=Sunday)", list(range(7)), index=2)
        classifier_name = st.selectbox("Classification Model", list(trained_classifiers.keys()))

    if st.button("Predict Traffic", type="primary"):
        # Make one row using the same column names used during training.
        input_values = {
            "Area Name": area,
            "Road/Intersection Name": road,
            "Traffic Volume": traffic_volume,
            "Average Speed": average_speed,
            "Travel Time Index": travel_time_index,
            "Road Capacity Utilization": road_capacity,
            "Environmental Impact": environmental_impact,
            "Public Transport Usage": public_transport,
            "Traffic Signal Compliance": signal_compliance,
            "Parking Usage": parking_usage,
            "Pedestrian and Cyclist Count": pedestrian_count,
            "Weather Conditions": weather,
            "Roadwork and Construction Activity": roadwork,
            "Month": month,
            "Day of Week": day
        }
        input_row = pd.DataFrame([input_values])
        simple_value = simple_model.predict(input_row[["Average Speed"]])[0]
        multiple_value = multiple_model.predict(input_row[regression_columns])[0]
        # Add the multiple-regression estimate for the classification model.
        class_input = input_row.copy()
        class_input["Congestion Level"] = multiple_value
        risk_value = trained_classifiers[classifier_name].predict(class_input[classification_columns])[0]

        a, b = st.columns(2)
        a.metric("Simple Linear Regression", f"{simple_value:.2f}")
        b.metric("Multiple Linear Regression", f"{multiple_value:.2f}")
        st.subheader("Predicted incident-count category")
        st.write("**" + risk_value + "**")
        
with tab3:
    st.subheader("Regression model evaluation")
    regression_results = pd.DataFrame([
        {"Model": "Simple Linear Regression", "MAE": mean_absolute_error(y_test_reg, simple_predictions), "RMSE": np.sqrt(mean_squared_error(y_test_reg, simple_predictions)), "R²": r2_score(y_test_reg, simple_predictions)},
        {"Model": "Multiple Linear Regression", "MAE": mean_absolute_error(y_test_reg, multiple_predictions), "RMSE": np.sqrt(mean_squared_error(y_test_reg, multiple_predictions)), "R²": r2_score(y_test_reg, multiple_predictions)}
    ])
    st.dataframe(regression_results.round(3), use_container_width=True)

    st.subheader("Classification model comparison")
    st.dataframe(pd.DataFrame(classifier_scores).sort_values("Accuracy", ascending=False).round(3), use_container_width=True)
    model_to_check = st.selectbox("View details for a classifier", list(trained_classifiers.keys()))
    model_predictions = trained_classifiers[model_to_check].predict(X_test_class)
    st.write("Classification report")
    st.dataframe(pd.DataFrame(classification_report(y_test_class, model_predictions, output_dict=True, zero_division=0)).transpose().round(3), use_container_width=True)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test_class, model_predictions, ax=ax, colorbar=False)
    ax.set_title(model_to_check + " confusion matrix")
    st.pyplot(fig)

    st.subheader("K-Fold validation for regression")
    fold_number = st.slider("Number of K-folds", 3, 10, 5)
    kfold_model = Pipeline([("prepare", regression_preparation), ("model", LinearRegression())])
    kfold = KFold(n_splits=fold_number, shuffle=True, random_state=42)
    kfold_scores = cross_val_score(kfold_model, X_reg, y_reg, cv=kfold, scoring="r2")
    st.write("Scores for each fold:", np.round(kfold_scores, 3).tolist())
    st.metric("Average K-fold R²", f"{kfold_scores.mean():.3f}")

    st.subheader("Stratified K-Fold validation for classification")
    stratified_number = st.slider("Number of Stratified folds", 3, 10, 5)
    stratified_model = Pipeline([("prepare", classification_preparation), ("model", RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1))])
    stratified = StratifiedKFold(n_splits=stratified_number, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(stratified_model, X_class, y_class, cv=stratified, scoring="accuracy")
    st.write("Scores for each fold:", np.round(stratified_scores, 3).tolist())
    st.metric("Average Stratified K-fold accuracy", f"{stratified_scores.mean():.3f}")

with tab4:
    st.subheader("Traffic pattern clustering")
    st.write("K-means and Gaussian Mixture Model group similar records using numerical traffic measurements.")
    cluster_columns = ["Traffic Volume", "Average Speed", "Travel Time Index", "Congestion Level", "Road Capacity Utilization", "Incident Reports", "Environmental Impact", "Public Transport Usage", "Traffic Signal Compliance", "Parking Usage", "Pedestrian and Cyclist Count"]
    cluster_data = df[cluster_columns].copy()
    cluster_data = cluster_data.fillna(cluster_data.median())
    scaled_cluster_data = StandardScaler().fit_transform(cluster_data)
    number_of_clusters = st.slider("Number of clusters", 2, 6, 3)
    method = st.radio("Clustering method", ["K-means", "Gaussian Mixture Model"], horizontal=True)
    if method == "K-means":
        cluster_model = KMeans(n_clusters=number_of_clusters, random_state=42, n_init=10)
        cluster_labels = cluster_model.fit_predict(scaled_cluster_data)
    else:
        cluster_model = GaussianMixture(n_components=number_of_clusters, random_state=42)
        cluster_labels = cluster_model.fit_predict(scaled_cluster_data)
    st.write("Records in each cluster")
    st.dataframe(pd.Series(cluster_labels).value_counts().sort_index().rename("Number of records"))
    x_column = st.selectbox("X-axis", cluster_columns, index=0)
    y_column = st.selectbox("Y-axis", cluster_columns, index=3)
    fig, ax = plt.subplots()
    for label in sorted(set(cluster_labels)):
        points = df.loc[np.array(cluster_labels) == label]
        ax.scatter(points[x_column], points[y_column], label="Cluster " + str(label), alpha=0.5)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(method + " clusters")
    ax.legend()
    st.pyplot(fig)
    cluster_summary = df[cluster_columns].copy()
    cluster_summary["Cluster"] = cluster_labels
    st.subheader("Average values in each cluster")
    st.dataframe(cluster_summary.groupby("Cluster")[cluster_columns].mean().round(2), use_container_width=True)

with tab5:
    st.subheader("Dataset preview")
    st.write("Rows:", len(df), "| Columns:", len(df.columns))
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button("Download dataset CSV", df.to_csv(index=False), "traffic_dataset.csv", "text/csv")

st.divider()
