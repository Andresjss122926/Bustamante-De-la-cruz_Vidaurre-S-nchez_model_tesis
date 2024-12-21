import os  # Operating system interfaces
import tensorflow as tf                                    # TensorFlow deep learning framework
import matplotlib.pyplot as plt                            # Plotting library
import matplotlib.image as mpimg                           # Image loading and manipulation library
from tensorflow.keras.models import Sequential, Model      # Sequential and Functional API for building models
from tensorflow.keras.optimizers import Adam               # Adam optimizer for model training
from tensorflow.keras.callbacks import EarlyStopping       # Early stopping callback for model training
from tensorflow.keras.regularizers import l1, l2           # L1 and L2 regularization for model regularization
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation and preprocessing for images
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  
# Various types of layers for building neural networks
from tensorflow.keras.applications import DenseNet121, EfficientNetB4, Xception, VGG16, VGG19   # Pre-trained models for transfer learning

# Definir las rutas de entrenamiento y validación
train_dir = '/home/user/tomatoesdetect/static/images/tomato/train'
val_dir = '/home/user/tomatoesdetect/static/images/tomato/val'
trained_model_path = '/home/user/tomatoesdetect/Trained_Model/modelo_entrenado.h5'

# Cargar los datos de entrenamiento
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
)

# Normalizar los datos de entrenamiento
train_data = train_data.map(lambda x, y: (x / 255.0, y))

# Cargar los datos de validación
val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
)

# Normalizar los datos de validación
val_data = val_data.map(lambda x, y: (x / 255.0, y))

print("Datos de entrenamiento y validación cargados correctamente.")

# directorio de imagenes
path = "/home/user/tomatoesdetect/static/images/tomato/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus"

# obtener lista de imagenes
image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Display the first 6 images with their labels and save the figure
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i in range(6):
    # Get the image file name and its label
    image_file = image_files[i]
    label = image_file.split('.')[0]

    # Load and display the image
    img_path = os.path.join(path, image_file)
    img = mpimg.imread(img_path)
    ax = axs[i // 3, i % 3]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label)

plt.tight_layout()
plt.savefig('/home/user/tomatoesdetect/visualization_output.png')
print("Visualización guardada en: /home/user/tomatoesdetect/visualization_output.png")

# Path to the directory containing images
path = "/home/user/tomatoesdetect/static/images/tomato/train/Tomato___Bacterial_spot"

# Get a list of all image file names in the directory
image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Display the first 6 images with their labels and save the figure
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i in range(6):
    # Get the image file name and its label
    image_file = image_files[i]
    label = image_file.split('.')[0]

    # Load and display the image
    img_path = os.path.join(path, image_file)
    img = mpimg.imread(img_path)
    ax = axs[i // 3, i % 3]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(label)

plt.tight_layout()
plt.savefig('/home/user/tomatoesdetect/visualization_bacterial_spot.png')
print("Visualización guardada en: /home/user/tomatoesdetect/visualization_bacterial_spot.png")

# Definir la base convolucional antes de agregarla al modelo
conv_base = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3),
    pooling='avg'
)

# Congelar los pesos de la base convolucional
conv_base.trainable = False

# Inicializar el modelo secuencial
model = Sequential()

# Agregar la base convolucional
model.add(conv_base)

# Añadir capas adicionales con BatchNormalization y Dropout
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(Dense(120, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Ajusta el número de clases según tu dataset

# Compilar el modelo (Paso 1)
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Modelo compilado y listo para entrenar.")

# Entrenar el modelo (Paso 2)
history = model.fit(
    train_data, 
    epochs=100, 
    validation_data=val_data, 
    callbacks=[EarlyStopping(patience=0)]
)

# Crear la carpeta si no existe y guardar el modelo entrenado
os.makedirs('/home/user/tomatoesdetect/Trained_Model', exist_ok=True)
model.save(trained_model_path)
print(f"Modelo entrenado y guardado en '{trained_model_path}'.")

# Evaluar el modelo en el conjunto de validación (Paso 3)
evaluation = model.evaluate(val_data)
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

