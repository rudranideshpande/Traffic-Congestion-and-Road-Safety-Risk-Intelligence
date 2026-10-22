# Traffic Congestion and Road Safety Risk Intelligence

## Run on Windows Command Prompt
1. Extract this folder.
2. Open CMD inside this folder.
3. Run:
   ```bat
   python -m venv venv
   venv\Scripts\activate
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

Keep `Banglore_traffic_Dataset.csv` in the same folder as `app.py`.

## Included syllabus topics
- EDA and one-hot encoded data preview
- Simple linear regression
- Multiple linear regression
- Logistic regression
- KNN classifier
- Decision tree
- Bagging
- Random forest classifier
- K-fold and stratified K-fold validation
- K-means clustering
- Gaussian mixture model

## Notes
The classification target is derived from the dataset's `Incident Reports` column: 2 or more reports is `Higher incident count`, otherwise `Lower incident count`. This is a proxy label, not a verified accident-risk probability. The `Incident Reports` column is not used as a classification input.
