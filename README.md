📚 Course Recommendation System 🚀
A Personalized Online Course Recommender System using Machine Learning.
This system suggests courses based on user preferences, enrollments, and content similarity using multiple recommendation techniques.

🌟 Project Overview
This project explores different recommender system approaches, including:
👉 Content-Based Filtering (TF-IDF + Cosine Similarity)
👉 Collaborative Filtering (KNN, NMF, Neural Network Embeddings)
👉 User Profile Clustering
👉 Performance Evaluation (Precision@K, Recall@K, RMSE)

🔗 GitHub Repository: git@github.com:bur3hani/Course-Recommendation-SYS.git

📂 Project Structure
Course-Recommendation-SYS/
│── data/                     # Dataset files
│── notebooks/                 # Jupyter Notebooks for model development
│── models/                    # Saved ML models
│── src/                       # Python scripts for recommendation algorithms
│   ├── content_based.py       # TF-IDF based recommendations
│   ├── collaborative_knn.py   # KNN-based collaborative filtering
│   ├── collaborative_nmf.py   # NMF-based collaborative filtering
│   ├── neural_networks.py     # Deep learning embeddings for recommendations
│── app/                       # Streamlit web application (if deployed)
│── requirements.txt           # Python dependencies
│── README.md                  # Project documentation
📊 Exploratory Data Analysis (EDA)
Course Distribution by Difficulty: Majority of courses are Beginner & Intermediate.
Most Popular Courses: Highly enrolled courses are mainly in AI, Data Science & Programming.
Word Cloud: Shows frequent keywords like "Machine Learning", "Python", "AI".
🛠️ Recommender System Implementations
1️⃣ Content-Based Filtering (TF-IDF)
Extracts text features from course descriptions.
Computes Cosine Similarity between courses.
Limitations: Does not consider user interactions.
tfidf = TfidfVectorizer(stop_words="english", max_features=500)
tfidf_matrix = tfidf.fit_transform(df["course_description"].fillna(""))
cosine_sim = cosine_similarity(tfidf_matrix)
2️⃣ KNN-Based Collaborative Filtering
Uses course enrollments to find similar courses.
KNN (k=6, cosine similarity) for recommendation.
Results: Precision@5 = 0.20, Recall@5 = 0.33.
knn = NearestNeighbors(n_neighbors=6, metric="cosine")
knn.fit(combined_features)
3️⃣ NMF-Based Collaborative Filtering
Uses Non-Negative Matrix Factorization (NMF) to find hidden user-course patterns.
Reduces user-course matrix into latent features.
Results: RMSE = 0.85 (better than KNN).
nmf = NMF(n_components=15, init="random", random_state=42)
user_matrix = nmf.fit_transform(user_course_matrix)
course_matrix = nmf.components_
4️⃣ Neural Network Embedding-Based Recommender
Uses deep learning embeddings to learn user-course relationships.
Fully connected layers predict user preferences.
Results: Best performing model with RMSE = 0.78.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_courses, output_dim=50, input_length=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_courses, activation="softmax")
])
📊 Performance Evaluation
Model	Precision@5	Recall@5	RMSE
TF-IDF	0.00	0.00	N/A
KNN	0.20	0.33	0.92
NMF	0.25	0.40	0.85
Neural Network	0.30	0.45	0.78
🎨 Flowcharts & Visualizations
👉 Content-Based Recommender System Flowchart
👉 KNN Collaborative Filtering Flowchart
👉 NMF-Based Collaborative Filtering Flowchart
👉 Neural Network Embedding-Based Recommender Flowchart

🚀 How to Run the Project
🔹 Option 1: Run Locally
git clone git@github.com:bur3hani/Course-Recommendation-SYS.git
cd Course-Recommendation-SYS
pip install -r requirements.txt
jupyter notebook
Run notebooks inside /notebooks/ to train and test models.

🔹 Option 2: Streamlit Web App (If Deployed)
Deployed App Link: Click Here (Replace with actual deployment URL)
🔥 Future Improvements
👉 Integrate User Behavior Data for personalization.
👉 Improve Course Recommendations with Transformers (BERT).
👉 Deploy as an Interactive Web App (Streamlit, Flask, or FastAPI).

💪 Contributing
Feel free to fork this repository, raise issues, and submit pull requests.
For major changes, please open an issue first to discuss what you would like to improve.

📄 License
This project is licensed under the MIT License.

👨‍💻 Author
Developed by @bur3hani 🚀

🔗 GitHub Repo: git@github.com:bur3hani/Course-Recommendation-SYS.git
📞 Contact: [http://buruops.com]
