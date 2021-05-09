from django.shortcuts import render, redirect

# Create your views here.
from pathlib import Path
from time import time

from django.core.files.storage import FileSystemStorage
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf


def home_view(request):
    return render(request, 'home.html')


def upload_view(request):
    base_path = Path(__file__).resolve().parent.parent
    path_model = base_path / 'models/checkpoint-141.hdf5'
    path_upload = base_path / 'media'
    path_processed = path_upload / 'processed_images'
    path_processed.mkdir(parents=True, exist_ok=True)
    batch_size = 64
    num_classes = 2
    no_epoch = 128
    init_lr = 0.001

    def process_image(input_path, output_path):
        img = np.array(Image.open(input_path), dtype=np.float32)
        img_255 = img / 255.0
        w, h, _ = img.shape
        v = 64
        r = w // v
        c = h // v
        data = np.zeros((r * c, v, v, 3), dtype=np.float32)
        idx = 0

        for i in range(r):
            for j in range(c):
                data[idx, :, :, :] = img_255[i * v : (i + 1) * v, j * v : (j + 1) * v, :]
                idx += 1

        mat = np.argmax(model.predict(data), axis=1).reshape((r, c))
        r, c = mat.shape

        for i in range(r):
            for j in range(c):
                if mat[i, j] == 0:
                    img[i * v : (i + 1) * v, j * v : (j + 1) * v, 2] = 0.0

        new_image = Image.fromarray(np.uint8(img), 'RGB')
        new_image.save(output_path)


    if request.method == 'POST' and 'photo' in request.FILES:
        image_file = request.FILES['photo']
        # print(image_file.name, image_file.size)
        image_name = image_file.name
        fss = FileSystemStorage()

        if Path.exists(path_upload / image_file.name):
            Path.unlink(path_upload / image_file.name)
            fss.save(image_file.name, image_file)
        else:
            fss.save(image_file.name, image_file)

        # with tf.device('/device:gpu:0'):
        start_time = time()

        inp_img = Input(shape=(64, 64, 3))
        model = Conv2D(32, (3, 3), activation='relu')(inp_img)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Conv2D(64, (3, 3), activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Conv2D(128, (3, 3), activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Conv2D(128, (3, 3), activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Flatten()(model)
        model = Dropout(0.5)(model)
        model = Dense(512, activation='relu')(model)
        model = Dense(1, activation='sigmoid')(model)
        out = Dense(num_classes, activation='softmax')(model)
        model = Model(inputs=inp_img, outputs=out)

        opt = Adam(lr=init_lr, decay=init_lr/no_epoch)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        model.load_weights(path_model)

        input_path = path_upload / image_name
        image_name = image_name.split('.')[0] + '_processed.jpg'
        output_path = path_processed / image_name
        process_image(input_path, output_path)

    context = {
        'filename': image_file.name,
        'filename_processed': image_name,
        'time': round(time() - start_time, 2),
    }

    return render(request, 'upload.html', context)
