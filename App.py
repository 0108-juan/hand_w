import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Configuración de la página de Streamlit
st.set_page_config(
    page_title='Reconocimiento de Dígitos Escritos a Mano',
    layout='wide',
    page_icon="✍️",
    initial_sidebar_state="expanded"
)

# Aplicar estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #2e86ab;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1668a3;
        transform: scale(1.05);
    }
    .canvas-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">✍️ Reconocimiento de Dígitos Escritos a Mano</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Dibuja un dígito en el panel y presiona "Predecir"</h2>', unsafe_allow_html=True)

# Layout con columnas
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Panel de Dibujo")
    
    # Controles de dibujo
    drawing_mode = "freedraw"
    stroke_width = st.slider('**Grosor del pincel**', 1, 30, 15, help="Ajusta el grosor del trazo para dibujar")
    stroke_color = '#FFFFFF'
    bg_color = '#000000'
    
    # Contenedor para el canvas centrado
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=280,
        width=280,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Botón de predicción
    if st.button('🔍 Predecir Dígito', use_container_width=True):
        if canvas_result.image_data is not None:
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
            input_image.save('prediction/img.png')
            img = Image.open("prediction/img.png")
            
            # Mostrar spinner mientras se procesa
            with st.spinner('Procesando tu dígito...'):
                res = predictDigit(img)
            
            # Mostrar resultado
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### ¡El dígito es: **{res}**!")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Mostrar confianza de predicción
            st.info("💡 **Consejo:** Para mejores resultados, dibuja el dígito centrado y con trazos claros.")
        else:
            st.error("⚠️ Por favor, dibuja un dígito en el canvas antes de predecir.")

with col2:
    st.markdown("### Información de la Aplicación")
    
    # Tarjeta informativa
    with st.expander("ℹ️ ¿Cómo funciona?", expanded=True):
        st.markdown("""
        Esta aplicación utiliza una **red neuronal convolucional (CNN)** entrenada con el 
        famoso dataset MNIST para reconocer dígitos escritos a mano.
        
        **Proceso:**
        1. Dibuja un dígito (0-9) en el panel izquierdo
        2. Ajusta el grosor del pincel si es necesario
        3. Haz clic en 'Predecir Dígito'
        4. ¡Observa el resultado!
        """)
    
    # Espacio para estadísticas o información adicional
    st.markdown("### Estadísticas del Modelo")
    st.markdown("""
    - **Precisión en pruebas:** >98%
    - **Dataset utilizado:** MNIST
    - **Arquitectura:** Red Neuronal Convolucional
    - **Entrada:** Imágenes 28x28 píxeles en escala de grises
    """)

# Barra lateral mejorada
with st.sidebar:
    st.markdown("## Acerca de")
    st.markdown("---")
    
    st.markdown("""
    Esta aplicación demuestra la capacidad de una **Red Neuronal Artificial (RNA)** 
    para reconocer dígitos escritos a mano.
    
    **Características:**
    - Interfaz intuitiva para dibujar dígitos
    - Reconocimiento en tiempo real
    - Alta precisión de predicción
    """)
    
    st.markdown("---")
    st.markdown("### Créditos")
    st.markdown("Basado en el desarrollo de **Vinay Uniyal**")
    
    # Enlace al repositorio (descomentar si tienes el enlace)
    # st.markdown("[🔗 Ver código en GitHub](https://github.com/Vinay2022/Handwritten-Digit-Recognition)")
    
    st.markdown("---")
    st.markdown("### Instrucciones Rápidas")
    st.markdown("""
    1. Usa el panel izquierdo para dibujar
    2. Ajusta el grosor si es necesario
    3. Presiona 'Predecir Dígito'
    4. ¡Listo!
    """)
