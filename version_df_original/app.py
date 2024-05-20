from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Charger le fichier CSV
water = pd.read_csv('water_potability.csv')

# Charger le modèle
model = joblib.load('best_lgb_model_v2.pkl')
scaler = joblib.load('scaler_v2.pkl')  # Charger RobustScaler

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        ph = float(request.form['ph'])
        Hardness = float(request.form['Hardness'])
        Solids = float(request.form['Solids'])
        Chloramines = float(request.form['Chloramines'])
        Sulfate = float(request.form['Sulfate'])
        Conductivity = float(request.form['Conductivity'])
        Organic_carbon = float(request.form['Organic_carbon'])
        Trihalomethanes = float(request.form['Trihalomethanes'])
        Turbidity = float(request.form['Turbidity'])
        
        # Préparer les données pour la prédiction
        data = [[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]
        
        # Mettre à l'échelle les données
        data_scaled = scaler.transform(data)  # Mettre à l'échelle avec RobustScaler
        
        # Faire la prédiction
        
        prediction = model.predict(data_scaled)[0]
        prediction_result = "Potable" if prediction == 1 else "Non Potable"
        
        
        # Afficher les histogrammes avec la valeur saisie
        img = io.BytesIO()
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 6))
        water.drop('Potability', axis=1).hist(ax=axes.flatten(), bins=30, color='royalblue', alpha=0.7)
        for i, ax in enumerate(axes.flatten()):
            ax.axvline(x=data[0][i], color='red', linestyle='--')
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return render_template('predict.html', prediction_result=prediction_result, plot_url=plot_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
