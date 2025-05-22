# London Bus Time Prediction

A machine learning-based system for predicting bus arrival times in London, UK. This project uses a Bidirectional LSTM neural network to provide accurate predictions for various bus routes.

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/EthanPisani/WDS_LTC_BUS.git
cd WDS_LTC_BUS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
python frontend.py
```

The application will open automatically in your default web browser. If it doesn't, access it at `http://localhost:8501`.

## 📚 Project Overview

This project consists of two main components:

1. **Backend API**
   - Built with Flask
   - Handles model predictions
   - Provides route and stop information
   - Serves at `https://ltc-bus-api.ethanpisani.com`

2. **Frontend Application**
   - Built with Streamlit
   - User-friendly interface for bus time predictions
   - Real-time updates and predictions
   - Model performance visualization

## 🛠️ Project Structure

```
WDS_LTC_BUS/
├── frontend.py            # Main Streamlit application
├── Model.py               # Neural network implementation
├── train_gpu.py          # Training script
├── eval2.py              # Evaluation script
├── requirements.txt      # Project dependencies
├── model.pth             # Trained model weights
├── stop_id_mapping.json  # Bus stop ID mappings
├── bus.csv               # Dataset
└── analysis_results/     # Model analysis and visualizations
```

## 📊 Features

- Real-time bus arrival predictions using a Bidirectional LSTM neural network
- Interactive route and stop selection interface
- Model performance visualization
- Save favorite routes for quick access
- Mobile-responsive design
- Detailed model architecture explanation

## 🛠️ Technologies Used

- **Machine Learning**
  - PyTorch
  - Scikit-learn
  - Pandas
  - NumPy

- **Web Frameworks**
  - Streamlit (Frontend)
  - Flask (Backend API)

- **Data Visualization**
  - Matplotlib
  - Seaborn
  - Plotly

## 📈 Model Architecture

The prediction model is a Bidirectional LSTM neural network that:
- Processes sequential data for accurate time predictions
- Handles both past and future context
- Uses feature scaling for optimal performance
- Includes dropout layers for regularization

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Ethan Pisani
- Henrique Leite
- Hadi Youssef
- Marc Alex Crasto
- Mohannad Salem
- Mollo Hou
- Riley Wong
- Saad Naeem
