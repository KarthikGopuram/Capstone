import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# Import LinearRegression from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# Load the state data
country_list = pickle.load(open('country_dict.pkl', 'rb'))
country = pd.DataFrame(country_list)

def predict(Country, Crops, Year):
    user_crop = Crops

    # Filter data for the selected country and crop
    user_data = country[(country['Area'] == Country) & (country['Item'] == user_crop)]

    # Split data into training and testing sets (You may need more advanced data splitting)
    X = user_data[['Year']]
    y = user_data['Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Receive user input for the year to make a prediction
    user_year = int(Year)

    # Predict production for the user-specified year
    predicted_yield = model.predict([[user_year]])[0]
    return predicted_yield

st.title('AGRICULTURE CROP PREDICTION USING TIME SERIES ANALYSIS')

Country = st.selectbox(
    "Select the state from the list",
    country['Area'].unique(),
    #index=None,
    #format_func=lambda x: x,
    #placeholder="Select state",
)

st.write('You selected state:', Country)

Crops = st.selectbox(
    "Select the Crop from the list",
    country[country['Area'] == Country]['Item'].unique(),
    #index=None,
    #format_func=lambda x: x,
    #placeholder="Select Crop",
)

st.write('You selected state:', Crops)

Year = st.text_input('Enter year', placeholder="Enter year")
st.write('Year:', Year)

if st.button('Predict'):
    prediction = predict(Country, Crops, Year)

    # country_data = country[country['Area'] == Country]
    # # Remove rows with missing 'Production' values
    # country_data = country_data.dropna(subset=['Value'])
    # # Extract the relevant columns
    # years = country_data['Year']
    # production = country_data['Value'].astype(float)
    # Filter data for the selected country and crop
    user_data = country[(country['Area'] == Country) & (country['Item'] == Crops)]

    # Split data into training and testing sets (You may need more advanced data splitting)
    X = user_data[['Year']]
    y = user_data['Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Receive user input for the year to make a prediction
    user_year = int(Year)

    # Predict production for the user-specified year
    predicted_yield = model.predict([[user_year]])[0]

    # Plot historical data and the predicted value as a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(user_data['Year'], user_data['Value'], label='Historical Data', color='blue', marker='o', linestyle='-')
    plt.plot(user_year, predicted_yield, label='Predicted Value', color='red', marker='o')
    plt.xlabel('Year')
    plt.ylabel('Yield (hg/ha)')
    plt.title(f'Crop Yield for {Crops} in {Country}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the plot as an image
    plt.savefig('production_plot.png')

    # Display the saved plot in Streamlit
    st.image('production_plot.png')

    st.write('Crop prediction for ' + str(Crops) + ' in the year ' + str(Year) + ' in ' + str(Country) + ' is ' + str(prediction))

# Function to predict production

