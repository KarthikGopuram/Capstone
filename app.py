import pickle

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import LinearRegression from scikit-learn
from sklearn.linear_model import LinearRegression

# Create an instance of the LinearRegression model
model = LinearRegression()

# Now you can use the 'model' instance to fit the model, make predictions, and perform other linear regression tasks.


def predict(State, year):
    Year = int(year)
    state_data = states[states['State'] == State]
    # Remove rows with missing 'Production' values
    state_data = state_data.dropna(subset=['Production'])
    # Extract the relevant columns
    years = state_data['Year'].str.split('-', expand=True)[0].astype(int)
    production = state_data['Production'].astype(float)
    # Create a linear regression model
    model = LinearRegression()
    model.fit(years.values.reshape(-1, 1), production.values.reshape(-1, 1))
    # Predict the production for the desired year
    predicted_production = model.predict([[Year]])[0][0]

    # Create a bar plot for the historical data
    plt.bar(years, production, label='Historical Data', color='blue')
    plt.xlabel('Year')
    plt.ylabel('Production (Tonnes)')
    plt.title(f'Production Trend for {State}')

    # Add a bar for the predicted value
    plt.bar(Year, predicted_production, color='red', label=f'Predicted {Year}', width=0.4)
    plt.legend()
    plt.grid(True)
    plt.show()
    return predicted_production

states_list = pickle.load(open('state_dict.pkl', 'rb'))
states = pd.DataFrame(states_list)
st.title('Crop prediction in INDIA')

State = st.selectbox(
   "Select the state from the list",
   (states['State'].unique()),
   index=None,
   placeholder="Select state",
)

st.write('You selected state:', State)


Year = st.text_input('Enter year', placeholder="Enter year")
st.write('Year:', Year)

if st.button('Predict'):
    prediction = predict(State, Year)
    st.write('Crop prediction of ' + str(State) + ' in the year ' + str(Year) + ' is ' + str(prediction))
