start cmd /k "streamlit run decision_tree.py --server.port 8501"
start cmd /k "streamlit run fraud_detection_app.py --server.port 8502"
python -m http.server 8000 --bind 127.0.0.1