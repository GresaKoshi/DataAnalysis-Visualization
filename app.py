import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="IoT Traffic ML App", layout="wide")
st.title("📡 IoT Traffic Prediction App")

st.markdown("""
Welcome to the IoT Traffic Machine Learning App.  
- Upload your **CSV data**.
- Select a model.
- View predictions and performance metrics.
""")

model_choice = st.selectbox("🧠 Choose a model for prediction:", ("Random Forest", "Decision Tree"))

try:
    if model_choice == "Random Forest":
        model = pickle.load(open("model1.pkl", "rb"))
    else:
        model = pickle.load(open("model2.pkl", "rb"))
except FileNotFoundError:
    st.error(f"❌ Could not load the selected model ({model_choice}). Make sure the .pkl file is in the same folder.")
    st.stop()

uploaded_file = st.file_uploader("📁 Upload new IoT traffic data (.csv)", type=["csv"])

if uploaded_file is not None:
    try:

        data = pd.read_csv(uploaded_file)
        if data.empty or data.columns.size == 0:
            st.error("⚠️ The uploaded CSV is empty or has no readable columns. Please check the file content.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to read the uploaded CSV: {e}")
        st.stop()

    try:
        for col in ['Timestamp', 'Src_IP', 'Dst_IP']:
            if col in data.columns:
                data.drop(columns=col, inplace=True)

        if 'Device_Type' in data.columns:
            device_map = {'Smart Lock': 0, 'Smart Camera': 1, 'Smart TV': 2, 'Smart Light': 3}
            data['Device_Type'] = data['Device_Type'].map(device_map)

        if 'Flags' in data.columns:
            flags_map = {'SYN': 0, 'ACK': 1, 'RST': 2, 'FIN': 3}
            data['Flags'] = data['Flags'].map(flags_map)

        if 'Protocol' in data.columns:
            protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
            data['Protocol'] = data['Protocol'].map(protocol_map)

        if 'Activity' in data.columns:
            activity_map = {
                'sending data': 0,
                'receiving data': 1,
                'idle': 2,
                'streaming': 3
            }
            data['Activity'] = data['Activity'].map(activity_map)

        data.fillna(0, inplace=True)

        expected_features = [
            'Device_Type', 'Flags', 'Protocol',
            'TTL', 'Src_Port', 'Dst_Port',
            'Packet_Length', 'Payload_Size'
        ]

        for col in expected_features:
            if col not in data.columns:
                data[col] = 0

        data = data[expected_features]

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        if st.button("🔍 Predict"):
            try:
                predictions = model.predict(data)
                data['Prediction'] = predictions

                st.subheader("🔮 Prediction Results")
                st.dataframe(data, use_container_width=True)

                st.subheader("📊 Prediction Distribution")
                fig1, ax1 = plt.subplots()
                sns.countplot(x='Prediction', data=data, ax=ax1)
                st.pyplot(fig1)

                full_data = pd.read_csv(uploaded_file)
                if 'Label' in full_data.columns:
                    st.subheader("📈 Model Performance")
                    full_data['Prediction'] = predictions

                    report = classification_report(full_data['Label'], full_data['Prediction'], output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                    cm = confusion_matrix(full_data['Label'], full_data['Prediction'])
                    fig2, ax2 = plt.subplots()
                    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax2)
                    st.pyplot(fig2)
                else:
                    st.info("🛈 Add a 'Label' column to your CSV to evaluate model performance.")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
    except Exception as e:
        st.error(f"❌ Error during preprocessing: {e}")
