import numpy as np
from keras.models import load_model



data = np.load('/Volumes/Data_For_Mac/Big_Data_Data/Rory_DATA/data/feature_models/reshap_to_npy/all_labeled_data_X.npy')
data_y = np.load('/Volumes/Data_For_Mac/Big_Data_Data/Rory_DATA/data/feature_models/reshap_to_npy/all_labeled_data_y.npy')



model_path = ""

encoder = load_model(model_path)

X_data = []
print(data)
for i in range(1,data.shape[0]):
    data_to_encode =  data[i:i+1][:][:]
    latent_space = encoder.predict(data_to_encode)
    flat_latent_space = latent_space.flatten()
    X_data.append(flat_latent_space)



Y_data = data_y



