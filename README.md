
# London Bus Time Prediction

A machine learning-based system for predicting bus arrival times in London, UK. This project uses a Bidirectional LSTM neural network to provide accurate predictions for various bus routes.

## 🚀 Quick Start (Frontend Only)

1. Clone the repository:

```bash
git clone https://github.com/EthanPisani/WDS_LTC_BUS.git
cd WDS_LTC_BUS/frontend
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the frontend:

```bash
streamlit run frontend.py
```

Access the app at `http://localhost:8501` in your browser.

---

## 🐳 Docker (Full Stack)

To run both backend and frontend services:

1. From the root of the repository:

```bash
docker-compose up --build
```

2. Services:

* **Backend** at `http://localhost:5240`
* **Frontend** at `http://localhost:5241`

---

## 📚 Project Overview

This project consists of two main components:

1. **Backend API**

   * Built with Flask
   * Handles model predictions
   * Provides route and stop information
   * Runs on `http://localhost:5240`

2. **Frontend Application**

   * Built with Streamlit
   * Interactive UI for bus time predictions
   * Runs on `http://localhost:5241`

---

## 🛠️ Project Structure

```
WDS_LTC_BUS/
├── backend/               # Flask app and model files
│   ├── app.py
│   ├── Model.py
│   ├── best_model.pth
│   ├── requirements.txt
│   └── ...
├── frontend/              # Streamlit app
│   ├── frontend.py
│   ├── requirements.txt
│   └── ...
├── docker-compose.yml     # Multi-container setup
└── README.md
```

---

## 📊 Features

* Real-time bus arrival predictions
* Interactive route/stop selector
* Model performance visualizations
* Mobile-responsive Streamlit frontend

---

## 🛠️ Technologies Used

* **Machine Learning**: PyTorch, Scikit-learn, Pandas, NumPy
* **Web Frameworks**: Flask (API), Streamlit (Frontend)
* **Visualization**: Matplotlib, Plotly

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

* Ethan Pisani
* Henrique Leite
* Hadi Youssef
* Marc Alex Crasto
* Mohannad Salem
* Mollo Hou
* Riley Wong
* Saad Naeem

```