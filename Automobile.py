import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('automobile.csv')
data['horsepower'] = pd.to_numeric(data['horsepower'], errors = 'coerce')

st.markdown("<h1 style = 'color: #FF204E; text-align: center; font-size: 60px; font-family: Georgia'>CAR PRICE PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #A0153E; text-align: center; font-family: italic'>BUILT BY FRANCES </h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)

# #add image
st.image('pngwing.com (10).png')

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)


st.markdown("<p>The primary objective of this machine learning project is to develop an accurate and robust predictive model for estimating the price of a car based on its various features. By leveraging advanced machine learning algorithms, the aim is to create a model that can analyze and learn from historical car data, encompassing attributes such as make, model, year, mileage, engine type, fuel efficiency, and other relevant parameters.</p>", unsafe_allow_html = True)

st.sidebar.image('pngwing.com (11).png',caption = 'Welcome User')


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)


st.sidebar.subheader('User Input Variables')

sel_cols = ['curb-weight','normalized-losses','make', 'body-style', 'horsepower', 'city-mpg',
            'height','price']


curb = st.sidebar.number_input('curb-weight', data['curb-weight'].min(), data['curb-weight'].max())
norm = st.sidebar.number_input('normalized-losses', data['normalized-losses'].min(), data['normalized-losses'].max())
make = st.sidebar.selectbox('make', data['make'].unique())
body= st.sidebar.selectbox('body-style', data['body-style'].unique())
horse = st.sidebar.number_input('horsepower', data['horsepower'].min(), data['horsepower'].max())
city = st.sidebar.number_input('city-mpg', data['city-mpg'].min(), data['city-mpg'].max())
height = st.sidebar.number_input('height', data['height'].min(), data['height'].max())



#users input
input_var = pd.DataFrame()
input_var['curb-weight'] = [curb]
input_var['normalized-losses'] = [norm]
input_var['make'] = [make]
input_var['body-style'] = [body]
input_var['horsepower'] = [horse]
input_var['city-mpg'] = [city]
input_var['height'] = [height]


st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)


# import the transformers
make = joblib.load('make_encoder.pkl')
body = joblib.load('body-style_encoder.pkl')
#horse = joblib.load('horsepower_encoder.pkl')




# transform the users input with the imported scalers
input_var['make'] = make.transform(input_var[['make']])
input_var['body-style'] = body.transform(input_var[['body-style']])
# input_var['horsepower'] = horse.transform(input_var[['horsepower']])

# st.header('Transformed Input Variable')
# st.dataframe(input_var, use_container_width = True)

#modelling
model = joblib.load('AutomobileModel.pkl')


#to have a button for the user
if st.button('Predict Price'):
    predicted_price = model.predict(input_var)
    st.success(f"The Price of this Car is  {predicted_price[0].round()}")
    





















