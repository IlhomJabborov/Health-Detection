import streamlit as st
from fastai.vision.all import *
import plotly.express as px




st.write("## X-Ray Sog'liq detektori")
st.write(
    ":hospital: Ko‘krak qafasi rentgenogrammasi yordamida COVID-19,Pneumoniaga chalingan va Sog'lom insonni ajratish :female-doctor:"
)
st.sidebar.write("## X-Ray Rasmini Yuklash :gear:")

col1, col2,col3 = st.columns(3)
file=st.sidebar.file_uploader("Rasm yuklash",type=['png','jpeg','svg','jpg'])



if file:
    col1.write("Yuklangan Rasm :camera:")
    with col1:
        st.image(file)

    img =PILImage.create(file)

    model = load_learner("covid_model.pkl")

    pred, pred_id, probs =model.predict(img)

    col2.write("Natijalar 🧬 🧠")
    with col2:
        st.success(f"Bashorat : {pred}")
        st.info(f"Ehtimollik : {probs[pred_id]*100:.1f}%")
        
    col3.write("Bashorat aniqlik Grafigi 📈")
    with col3:
        fig = px.bar(y=probs*100,x=model.dls.vocab,width=400, height=350)
        st.plotly_chart(fig)
else:
    col1.write("Yuklangan Rasm :camera:")
    with col1:
        st.image("x_ray.jpg")
    col2.write("Natijalar 🧬 🧠")
    with col2:
        st.success(f"Bashorat Qiymati")
        st.info(f"Bashorat aniqligi (%)")
    col3.write("Bashorat aniqlik Grafigi 📈")
    with col3:
        st.image("grafik.jpg")
      

with st.expander("Foydalanish Qo'llanmasi"):
    st.markdown("""
                ##### Foydalanish:
                * Ekraning chap tomonidagi "Rasm yuklash" qismi orqali rentgenogramma(X-Ray) rasmingizni yuklang
                * Natijalar : Matn,Foiz va Grafik ko'rinishida chiqadi
                """)
with st.expander("Sayt haqida"):
    st.markdown("""
                ##### Vazifa :
                Mening sun'iy intellekt saytim, X-Ray rasmlar orqali covid-19, pnevmoniya va sog'lom insonni ajrata oladigan innovatsion texnologiyaga asoslangan. Bu sayt, tibbiy tahlil va tashhis jarayonlarini yanada tezroq va samarali qilish maqsadida yaratilgan.

                ##### Maqsad :
                Mening maqsadim, xronik tibbiy muammolar va pandemiya kabi jiddiy vaziyatlarda tibbiy tahlil va tekshirishlarni tez va samarali tashkil qilish orqali dunyo bo'ylab sog'lomlikni oshirishga yordam berishdir.
                """)
with st.expander("Aloqa"):
    st.markdown("""
                ##### Shaxsiy chat :
                * [Telegram](https://t.me/MY_F4TH3R)

                ##### Ijtimoiy tarmoqlarda ham kuzating :
                * [GitHub](https://github.com/JabborovRoboCoder)
                * [Telegram Channel](https://t.me/Only_IT_blog_Ilhom_Jabborov)
                * [You Tube Channel](https://www.youtube.com/channel/UCSYhTII7NUg1rCFfl2F7Gwg)

                """)




