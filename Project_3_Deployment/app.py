from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import datetime

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        
        try:
            store_nbr = int(request.form['store_nbr'])
            onpromotion = int(request.form['onpromotion'])
            year = int(request.form['year'])
            month = int(request.form['month'])
            day = int(request.form['day'])
            is_holiday = int(request.form['is_holiday'])

            # Validate date
            date_obj = datetime.date(year, month, day)
            weekday = date_obj.weekday()

            input_data = pd.DataFrame(
                [[store_nbr, onpromotion, year, month, day, weekday, is_holiday]],
                columns=['store_nbr', 'onpromotion', 'year', 'month', 'day', 'weekday', 'is_holiday']
            )

            prediction = model.predict(input_data)[0]

        except ValueError:
            error = "Invalid date selected. Please choose a valid calendar date."

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == '__main__':
    app.run(debug=True)
