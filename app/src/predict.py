from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

def predict():
    Brand = request.form.get('Brand')
    Dial_Shape = request.form.get('Dial_Shape')
    Strap_Color = request.form.get('Strap_Color')
    Strap_Material = request.form.get('Strap_Material')
    Touchscreen = request.form.get('Touchscreen')
    Model_Name = request.form.get('Model_Name')
    Current_Price = float(request.form.get('Current_Price'))
    Original_Price = float(request.form.get('Original_Price'))
    Rating = float(request.form.get('Rating'))
    Number_of_Ratings = float(request.form.get('Number_of_Ratings'))
    Battery_Life = float(request.form.get('Battery_Life'))
    Display_Size = float(request.form.get('Display_Size'))
    Weight = float(request.form.get('Weight'))

    numerical_data = [Current_Price, Original_Price, Rating, Number_of_Ratings, Battery_Life, Display_Size, Weight]
    categorical_data = [Brand, Dial_Shape, Strap_Color, Strap_Material, Touchscreen, Model_Name]
    data = numerical_data + categorical_data

    numerical_col = ['Current_Price', 'Original_Price', 'Rating', 'Number_of_Ratings', 'Battery_Life', 'Display_Size', 'Weight']
    cat_col = ['Brand', 'Dial_Shape', 'Strap_Color', 'Strap_Material', 'Touchscreen', 'Model_Name']
    col = numerical_col + cat_col

    df1 = pd.DataFrame(dict(zip(col, data)), index=[0])
    print(numerical_data, categorical_data)
    print(df1)

    model = pickle.load(open(r'src\models\best_model.pkl', "rb"))
    cat_encoder = pickle.load(open(r'src\models\cat_encoder.pkl', "rb"))
    scalar = pickle.load(open(r'src\models\scalar.pkl', "rb"))

    df1[cat_col] = cat_encoder.transform(df1[cat_col])
    print(df1)
    data_scaled = scalar.transform([df1.loc[0].values])
    result = model.predict(data_scaled)

    return result.round(2)