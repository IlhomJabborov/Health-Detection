import streamlit as st
from fastai.vision.all import *
import plotly.express as px

st.title('Salom')

file=st.file_uploader("Rasm yuklash",type=['png','jpeg','svg','jpg'])

if file:
    st.image(file)

    img =PILImage.create(file)

    model = load_learner("covid_model.pkl")

    pred, pred_id, probs =model.predict(img)
    st.success(f"Bashorat : {pred}")
    st.info(f"Ehtimollik : {probs[pred_id]*100:.1f}%")

    fig = px.bar(y=probs*100,x=model.dls.vocab)
    st.plotly_chart(fig)
