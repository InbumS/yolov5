import streamlit as st
import pandas as pd
import torch
import wget
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import time,sys

cfg_model_path = "models/best.pt" 

cfg_enable_url_download = True
if cfg_enable_url_download:
    url = "https://archive.org/download/bestweight_nulljoaheae/best.pt" #Configure this if you set cfg_enable_url_download to True
    cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
## END OF CFG


# default page setting
st.set_page_config(
    page_title="NULL Joahae",
    layout="wide"
)



def imageInput(device, src):
        
        if src=='Upload data':
                image_file=st.file_uploader("Uploaded An Image", type=['jpg','png','jpeg'])
                col1,col2=st.columns(2)
                if image_file is not None:
                        img=Image.open(image_file)
                        with col1:
                                st.image(img,caption='이미지 삽입', use_column_width='always')
                        ts=datetime.timestamp(datetime.now())
                        imgpath=os.path.join('data/uploads',str(ts)+image_file.name)
                        outputpath=os.path.join('data/outputs', os.path.basename(imgpath))
                        with open(imgpath, mode="wb") as f:
                                f.write(image_file.getbuffer())

                        model=torch.hub.load('ultralytics/yolov5','custom',path='models/best.pt',force_reload=True)
                        model.cuda() if device=='cuda'else model.cpu()
                        pred=model(imgpath)
                        pred.render() # rendering bbox in image
                        for im in pred.ims:
                                im_base64 =Image.fromarray(im) # fromarray: numpy array to img file
                                im_base64.save(outputpath)

                        # -- Display--
                        img_=Image.open(outputpath)
                        with col2:
                                st.image(img_,caption='예측 결과(Prediction)', use_column_width='always')
        

        elif src=='From dataset':
                test_images = os.listdir('data/images/')
                test_image = st.selectbox('Please select a test image:', test_images)
                image_file = 'data/images/' + test_image
                submit = st.button(" predict!")
                col1, col2 = st.columns(2)
                with col1:
                    img = Image.open(image_file)
                    st.image(img, caption='Selected Image', use_column_width='always')
                with col2:            
                    if image_file is not None and submit:
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True) 
                        pred = model(image_file)
                        pred.render()  # render bbox in image
                        for im in pred.ims:
                            im_base64 = Image.fromarray(im)
                            im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                             #--Display 
                            img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                            st.image(img_, caption='Model Prediction(s)')

def main():
     # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['Upload data', 'From dataset'])
        
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar
    st.header("PCB Image Defect Finder")
    st.subheader('👈🏽 왼쪽에서 옵션을 선택하세요!')
    st.sidebar.markdown("https://github.com/NULL-Joahae")
    imageInput(deviceoption, datasrc)
   

if __name__ =='__main__':     
    main()

# Download Model from url
@st.cache
def loadModel():
      start_dl=time.time()
      model_file=wget.download(url,out="models/")
      finished_dl=time.time()
      print(f"Model downloaded, ETA:{finished_dl-start_dl}")

if cfg_enable_url_download:
      loadModel()