# CAPSTONE
!pip install -q kaggle
!mkdir ~/content
!cp kaggle.json ~/content
!chmod 600 ~/content/kaggle.json
!kaggle datasets list
Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /root/.kaggle/kaggle.js
ref title size la
-------------------------------------------------------------------- -------------------------------------------------- ----- --
arnabchaki/data-science-salaries-2023 Data Science Salaries 2023 💸 25KB 2
tawfikelmetwally/automobile-dataset Car information dataset 6KB 20
fatihb/coffee-quality-data-cqi Coffee Quality Data (CQI May-2023) 22KB 20
mohithsairamreddy/salary-data Salary_Data 17KB 20
mauryansshivam/netflix-ott-revenue-and-subscribers-csv-file Netflix OTT Revenue and Subscribers (CSV File) 2KB 20
omarsobhy14/mcdonalds-revenue 🍟💰From Flipping Burgers to Billions: McDonald's 565B 
zsinghrahulk/rice-pest-and-diseases Rice - Pest and Diseases 312KB 20
iammustafatz/diabetes-prediction-dataset Diabetes prediction dataset 734KB 20
vstacknocopyright/fruit-and-vegetable-prices Fruit and Vegetable Prices 1KB 20
bilalwaseer/microsoft-stocks-from-1986-to-2023 Microsoft Stocks from 1986 to 2023 120KB 20
darshanprabhu09/stock-prices-for Stock prices of Amazon , Microsoft , Google, Apple 85KB 20
rajkumarpandey02/2023-world-population-by-country World Population by Country 38KB 20
danishjmeo/karachi-housing-prices-2023 Karachi_Housing_Prices_2023 1MB 20
adityaramachandran27/world-air-quality-index-by-city-and-coordinates World Air Quality Index by City and Coordinates 372KB 20
dansbecker/melbourne-housing-snapshot Melbourne Housing Snapshot 451KB 20
pushpakhinglaspure/oscar-dataset Oscar Academy Award-winning films 1927-2022 161KB 20
aryansingh0909/weekly-patent-application-granted Patent Application Granted Dataset 6MB 20
utkarshx27/heart-disease-diagnosis-dataset Heart Disease Prediction Dataset 3KB 20
shreyanshverma27/water-quality-testing Water Quality Testing 4KB 20
desalegngeb/conversion-predictors-of-cis-to-multiple-sclerosis Multiple Sclerosis Disease 3KB 20
!kaggle datasets download -d 'zeyadkhalid/faceshape-processed'
Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /root/.kaggle/kaggle.js
Downloading faceshape-processed.zip to /content
 98% 78.0M/79.8M [00:05<00:00, 21.8MB/s]
100% 79.8M/79.8M [00:05<00:00, 15.4MB/s]
import zipfile
zip_file = zipfile.ZipFile('/content/faceshape-processed.zip','r')
zip_file.extractall('/tmp')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_generator = ImageDataGenerator(rescale=1/255,
horizontal_flip=True,
vertical_flip=True,
rotation_range=60,
zoom_range=0.3,
fill_mode='nearest')
train_data = train_generator.flow_from_directory('/tmp/dataset/train',
target_size=(224,224),
batch_size=400,
class_mode='categorical',)
Found 3981 images belonging to 5 classes.
val_generator = ImageDataGenerator(rescale=1/255
)
val_data = train_generator.flow_from_directory('/tmp/dataset/test',
target_size=(224,224),
batch_size=100,
class_mode='categorical',)
Found 998 images belonging to 5 classes.
6/8/23, 11:13 AM Capston Trial.ipynb - Colaboratory
https://colab.research.google.com/drive/1mUgcwihoVDgstGlYaHyaRX4mw2epK-ci#scrollTo=XKV1PekQLsKr&printMode=true 2/4
# Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential([
Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
MaxPooling2D(2, 2),
Conv2D(32, (3,3), activation='relu'),
MaxPooling2D(3, 3),
Flatten(),
Dense(128, activation='relu'),
Dense(128, activation='relu'),
Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Training Model
history = model.fit(train_data,
steps_per_epoch=10,
epochs=50,
validation_data=val_data,
validation_steps=10,
verbose=2)
Epoch 22/50
10/10 - 57s - loss: 1.5173 - accuracy: 0.3225 - val_loss: 1.5330 - val_accuracy: 0.3216 - 57s/epoch - 6s/step
Epoch 23/50
10/10 - 57s - loss: 1.5238 - accuracy: 0.3238 - val_loss: 1.5357 - val_accuracy: 0.3417 - 57s/epoch - 6s/step
Epoch 24/50
10/10 - 57s - loss: 1.5147 - accuracy: 0.3401 - val_loss: 1.5201 - val_accuracy: 0.3176 - 57s/epoch - 6s/step
Epoch 25/50
10/10 - 57s - loss: 1.5175 - accuracy: 0.3235 - val_loss: 1.5297 - val_accuracy: 0.3116 - 57s/epoch - 6s/step
Epoch 26/50
10/10 - 57s - loss: 1.5233 - accuracy: 0.3180 - val_loss: 1.5366 - val_accuracy: 0.3166 - 57s/epoch - 6s/step
Epoch 27/50
10/10 - 57s - loss: 1.5183 - accuracy: 0.3215 - val_loss: 1.5236 - val_accuracy: 0.3427 - 57s/epoch - 6s/step
Epoch 28/50
10/10 - 58s - loss: 1.5062 - accuracy: 0.3288 - val_loss: 1.5008 - val_accuracy: 0.3467 - 58s/epoch - 6s/step
Epoch 29/50
10/10 - 57s - loss: 1.5113 - accuracy: 0.3336 - val_loss: 1.5526 - val_accuracy: 0.2856 - 57s/epoch - 6s/step
Epoch 30/50
10/10 - 57s - loss: 1.5058 - accuracy: 0.3389 - val_loss: 1.5352 - val_accuracy: 0.3166 - 57s/epoch - 6s/step
Epoch 31/50
10/10 - 57s - loss: 1.5009 - accuracy: 0.3451 - val_loss: 1.5120 - val_accuracy: 0.3297 - 57s/epoch - 6s/step
Epoch 32/50
10/10 - 58s - loss: 1.4886 - accuracy: 0.3439 - val_loss: 1.5187 - val_accuracy: 0.3407 - 58s/epoch - 6s/step
Epoch 33/50
10/10 - 57s - loss: 1.5033 - accuracy: 0.3283 - val_loss: 1.5157 - val_accuracy: 0.3477 - 57s/epoch - 6s/step
Epoch 34/50
10/10 - 58s - loss: 1.4879 - accuracy: 0.3409 - val_loss: 1.5168 - val_accuracy: 0.3257 - 58s/epoch - 6s/step
Epoch 35/50
10/10 - 57s - loss: 1.4909 - accuracy: 0.3426 - val_loss: 1.5143 - val_accuracy: 0.3297 - 57s/epoch - 6s/step
Epoch 36/50
10/10 - 56s - loss: 1.4877 - accuracy: 0.3592 - val_loss: 1.5037 - val_accuracy: 0.3277 - 56s/epoch - 6s/step
Epoch 37/50
10/10 - 57s - loss: 1.4779 - accuracy: 0.3477 - val_loss: 1.4968 - val_accuracy: 0.3287 - 57s/epoch - 6s/step
Epoch 38/50
10/10 - 58s - loss: 1.4635 - accuracy: 0.3670 - val_loss: 1.5161 - val_accuracy: 0.3527 - 58s/epoch - 6s/step
Epoch 39/50
10/10 - 56s - loss: 1.4634 - accuracy: 0.3662 - val_loss: 1.4715 - val_accuracy: 0.3507 - 56s/epoch - 6s/step
Epoch 40/50
10/10 - 58s - loss: 1.4610 - accuracy: 0.3760 - val_loss: 1.4917 - val_accuracy: 0.3307 - 58s/epoch - 6s/step
Epoch 41/50
10/10 - 58s - loss: 1.4736 - accuracy: 0.3534 - val_loss: 1.5055 - val_accuracy: 0.3457 - 58s/epoch - 6s/step
Epoch 42/50
10/10 - 56s - loss: 1.4743 - accuracy: 0.3562 - val_loss: 1.4749 - val_accuracy: 0.3557 - 56s/epoch - 6s/step
Epoch 43/50
10/10 - 57s - loss: 1.4594 - accuracy: 0.3708 - val_loss: 1.4880 - val_accuracy: 0.3367 - 57s/epoch - 6s/step
Epoch 44/50
10/10 - 57s - loss: 1.4754 - accuracy: 0.3615 - val_loss: 1.5375 - val_accuracy: 0.3206 - 57s/epoch - 6s/step
Epoch 45/50
10/10 - 56s - loss: 1.4762 - accuracy: 0.3574 - val_loss: 1.5110 - val_accuracy: 0.3437 - 56s/epoch - 6s/step
Epoch 46/50
10/10 - 57s - loss: 1.4705 - accuracy: 0.3723 - val_loss: 1.5037 - val_accuracy: 0.3457 - 57s/epoch - 6s/step
Epoch 47/50
10/10 - 57s - loss: 1.4562 - accuracy: 0.3745 - val_loss: 1.4837 - val_accuracy: 0.3577 - 57s/epoch - 6s/step
Epoch 48/50
10/10 - 56s - loss: 1.4603 - accuracy: 0.3677 - val_loss: 1.4837 - val_accuracy: 0.3677 - 56s/epoch - 6s/step
Epoch 49/50
10/10 - 57s - loss: 1.4443 - accuracy: 0.3713 - val_loss: 1.4711 - val_accuracy: 0.3747 - 57s/epoch - 6s/step
Epoch 50/50
10/10 - 57s - loss: 1.4494 - accuracy: 0.3791 - val_loss: 1.4798 - val_accuracy: 0.3517 - 57s/epoch - 6s/step
6/8/23, 11:13 AM Capston Trial.ipynb - Colaboratory
https://colab.research.google.com/drive/1mUgcwihoVDgstGlYaHyaRX4mw2epK-ci#scrollTo=XKV1PekQLsKr&printMode=true 3/4
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Nilai loss pada setiap epoch')
plt.legend(['training','validation'], loc='upper right')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Nilai accuracy pada setiap epoch')
plt.legend(['training','validation'], loc='upper right')
plt.show()

https://colab.research.google.com/drive/1mUgcwihoVDgstGlYaHyaRX4mw2epK-ci#scrollTo=XKV1PekQLsKr&printMode=true 4/4
