from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .forms import UserRegistrationForm
from .models import UserRegistrationModel

import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape


# ===============================
# USER REGISTRATION
# ===============================
def UserRegisterActions(request):

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)

        if form.is_valid():
            form.save()
            messages.success(request, "Registered Successfully")
            form = UserRegistrationForm()

        else:
            messages.success(request, "Email or Mobile already exists")

    else:
        form = UserRegistrationForm()

    return render(request, 'UserRegistrations.html', {'form': form})


# ===============================
# USER LOGIN
# ===============================
def UserLoginCheck(request):

    if request.method == "POST":

        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')

        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid,
                password=pswd
            )

            if check.status == "activated":

                request.session['id'] = check.id
                request.session['name'] = check.name

                return render(request, "users/UserHome.html")

            else:
                messages.success(request, "Account not activated")

        except:
            messages.success(request, "Invalid Login Details")

    return render(request, "UserLogin.html")


# ===============================
# USER HOME
# ===============================
def UserHome(request):
    return render(request, "users/UserHome.html")


# ===============================
# AUDIO PREPROCESS
# ===============================
def preprocess_audio(file_path):

    SAMPLE_RATE = 22050
    DURATION = 5
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512

    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    log_mel = librosa.power_to_db(mel_spec)

    log_mel = (log_mel - np.mean(log_mel)) / np.std(log_mel)

    return log_mel.T


# ===============================
# LOAD DATASET
# ===============================
def load_dataset(base_path):

    CLASSES = ['fan', 'pump', 'valve', 'slider']

    X = []
    y = []
    is_normal = []

    for class_name in CLASSES:

        train_path = os.path.join(base_path, class_name, "train")
        test_path = os.path.join(base_path, class_name, "test")

        if os.path.exists(train_path):

            for file in os.listdir(train_path):

                if file.endswith(".wav"):

                    path = os.path.join(train_path, file)

                    X.append(preprocess_audio(path))
                    y.append(class_name)
                    is_normal.append(1)

        if os.path.exists(test_path):

            for file in os.listdir(test_path):

                if file.endswith(".wav"):

                    path = os.path.join(test_path, file)

                    X.append(preprocess_audio(path))
                    y.append(class_name)

                    if "normal" in file:
                        is_normal.append(1)
                    else:
                        is_normal.append(0)

    return np.array(X), np.array(y), np.array(is_normal)


# ===============================
# BUILD MODEL
# ===============================
def build_model(input_shape, num_classes):

    inputs = Input(shape=input_shape)

    encoded = LSTM(64, return_sequences=True)(inputs)
    encoded = Dropout(0.2)(encoded)

    encoded = LSTM(32)(encoded)
    encoded = Dropout(0.2)(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)

    decoded = Dense(input_shape[0] * input_shape[1])(decoded)

    decoded = Reshape(input_shape, name="reconstruction")(decoded)

    classified = Dense(16, activation='relu')(encoded)

    classified = Dense(num_classes, activation='softmax', name="classification")(classified)

    model = Model(inputs=inputs, outputs=[decoded, classified])

    model.compile(

        optimizer="adam",

        loss={
            "classification": "sparse_categorical_crossentropy",
            "reconstruction": "mse"
        },

        loss_weights={
            "classification": 1.0,
            "reconstruction": 0.5
        },

        metrics={
            "classification": "accuracy"
        }
    )

    return model


# ===============================
# TRAIN MODEL
# ===============================
def training(request):

    print("\n==============================")
    print("🚀 Training Started...")
    print("==============================")

    CLASSES = ['fan', 'pump', 'valve', 'slider']

    base_path = os.path.join(settings.MEDIA_ROOT, "dataset")
    print("📂 Dataset Path:", base_path)

    # Load dataset
    print("\n📥 Loading Dataset...")
    X, y, is_normal = load_dataset(base_path)

    print("✅ Dataset Loaded Successfully")
    print("Total Samples:", len(X))

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("🔢 Encoding Labels Completed")

    # Train Test Split
    print("\n📊 Splitting Dataset (80% train / 20% test)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    # Normalize
    print("\n⚙ Normalizing Dataset")

    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test = (X_test - np.mean(X_test)) / np.std(X_test)

    print("✅ Normalization Completed")

    input_shape = (X_train.shape[1], X_train.shape[2])
    print("\n📐 Input Shape:", input_shape)

    # Build Model
    print("\n🧠 Building LSTM Model...")
    model = build_model(input_shape, len(CLASSES))

    print("✅ Model Created Successfully")

    # Train Model
    print("\n🏋 Training Model...")

    history = model.fit(
        X_train,
        [X_train, y_train],
        epochs=5,
        batch_size=32,
        validation_split=0.2
    )

    print("\n✅ Model Training Completed")

    # Save Model
    model_path = os.path.join(settings.MEDIA_ROOT, "anomaly_detection_model")

    model.save(model_path)

    print("💾 Model Saved at:", model_path)

    # Reconstruction error
    print("\n📊 Calculating Reconstruction Error")

    reconstructions = model.predict(X_train)[0]

    train_errors = np.mean(np.square(X_train - reconstructions), axis=(1,2))

    threshold = np.percentile(train_errors, 95)

    print("⚠ Reconstruction Threshold:", threshold)

    np.save(
        os.path.join(settings.MEDIA_ROOT, "reconstruction_error_threshold.npy"),
        threshold
    )

    print("💾 Threshold Saved")

    print("\n==============================")
    print("🎉 Training Completed Successfully")
    print("==============================\n")

    return render(request, "users/training.html", {"threshold": threshold})
# ===============================
# AUDIO PREDICTION
# ===============================

import os
from django.http import HttpResponse

import os
from django.http import HttpResponse

import os
from django.http import HttpResponse
import os
from django.http import HttpResponse

def rename_dataset(request):

    dataset_path = r"media\dataset"

    machines = ["fan", "pump", "slider", "valve"]

    total = 0

    print("\n===== DATASET RENAME START =====")

    for machine in machines:

        machine_path = os.path.join(dataset_path, machine)

        if not os.path.exists(machine_path):
            print("Machine folder missing:", machine_path)
            continue

        for sub in ["train", "test"]:

            sub_path = os.path.join(machine_path, sub)

            if not os.path.exists(sub_path):
                continue

            print("Scanning:", sub_path)

            count = 1

            for file in os.listdir(sub_path):

                if file.endswith(".wav"):

                    old_path = os.path.join(sub_path, file)

                    new_name = f"{machine}_{count}.wav"

                    new_path = os.path.join(sub_path, new_name)

                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed {file} → {new_name}")
                        total += 1
                    except Exception as e:
                        print("Rename error:", e)

                    count += 1

    print("===== RENAME FINISHED =====")
    print("Total files renamed:", total)

    return HttpResponse(f"Renaming completed. {total} files renamed.")

    dataset_path = r"C:\Users\STEEV\Downloads\36 anomaly_sound_detection\anomolib_sound_detection1\media\dataset"

    machines = ["fan", "pump", "slider", "valve"]

    total = 0

    print("\n===== DATASET RENAME START =====")

    for machine in machines:

        machine_path = os.path.join(dataset_path, machine)

        if not os.path.exists(machine_path):
            continue

        for subfolder in ["train", "test"]:

            folder_path = os.path.join(machine_path, subfolder)

            if not os.path.exists(folder_path):
                continue

            print("Checking folder:", folder_path)

            count = 1

            for file in os.listdir(folder_path):

                if file.lower().endswith(".wav"):

                    old_path = os.path.join(folder_path, file)

                    new_name = f"{machine}_{count}.wav"

                    new_path = os.path.join(folder_path, new_name)

                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed {file} → {new_name}")
                        total += 1
                    except Exception as e:
                        print("Error:", e)

                    count += 1

    print("===== RENAME FINISHED =====")
    print("Total files renamed:", total)

    return HttpResponse(f"Renaming completed. {total} files renamed.")
import librosa
import numpy as np
def preprocess_audio(file_path):

    # Load audio
    audio, sr = librosa.load(file_path, sr=22050)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    # Normalize MFCC
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    # Fix time length = 100
    if mfcc.shape[1] < 100:
        pad = 100 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :100]

    return mfcc
import os
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from tensorflow.keras.models import load_model
import os
def predict_audio(request):

    if request.method == "POST" and request.FILES.get("audio"):

        audio = request.FILES["audio"]

        # Save uploaded audio
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "audio"))
        filename = fs.save(audio.name, audio)

        file_path = os.path.join(settings.MEDIA_ROOT, "audio", filename)
        audio_url = fs.url(filename)

        # Preprocess audio
        processed = preprocess_audio(file_path)
        processed = processed.reshape(1, 13, 100)

        # Load model
        model_path = os.path.join(settings.MEDIA_ROOT, "audio_prediction.h5")
        model = load_model(model_path)

        # Load threshold
        threshold_path = r'reconstruction_error_threshold.npy'
        threshold = float(np.load(threshold_path))

        # Model prediction
        reconstructed = model.predict(processed)

        # Reconstruction error
        mse = np.mean(np.square(processed - reconstructed))

        # Detect anomaly
        if mse > threshold:
            result = "⚠️ Anomaly Sound Detected"
        else:
            result = "✅ Normal Machine Sound"

        # Detect machine category from filename
        name = filename.lower()

        if "fan" in name:
            predicted_class = "Fan"
        elif "pump" in name:
            predicted_class = "Pump"
        elif "valve" in name:
            predicted_class = "Valve"
        elif "slider" in name:
            predicted_class = "Slider"
        else:
            predicted_class = "Unknown Machine"

        return render(request, "users/UploadForm.html", {

            "predicted_class": predicted_class,
            "error": round(mse, 6),
            "result": result,
            "audio_url": audio_url

        })

    return render(request, "users/UploadForm.html")

    if request.method == "POST" and request.FILES.get("audio"):

        audio = request.FILES["audio"]

        # Save uploaded file
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "audio"))
        filename = fs.save(audio.name, audio)

        file_path = os.path.join(settings.MEDIA_ROOT, "audio", filename)
        audio_url = fs.url(filename)

        # Preprocess audio
        processed = preprocess_audio(file_path)
        processed = processed.reshape(1, 13, 100)

        # Load model
        model_path = os.path.join(settings.MEDIA_ROOT, "audio_prediction.h5")
        model = load_model(model_path)

        # Load threshold value
        threshold_path = r'reconstruction_error_threshold.npy'
        threshold = float(np.load(threshold_path))

        # Predict reconstruction
        reconstructed = model.predict(processed)

        # Calculate reconstruction error
        mse = np.mean(np.square(processed - reconstructed))

        # Detect anomaly
        if mse > threshold:
            result = "⚠️ Anomaly Sound Detected"
        else:
            result = "✅ Normal Machine Sound"

        # Detect machine type from filename
        name = filename.lower()

        if "fan" in name:
            predicted_class = "Fan"
        elif "pump" in name:
            predicted_class = "Pump"
        elif "valve" in name:
            predicted_class = "Valve"
        elif "slider" in name:
            predicted_class = "Slider"
        else:
            predicted_class = "Unknown Machine"

        return render(request, "users/UploadForm.html", {

            "predicted_class": predicted_class,
            "error": round(mse, 6),
            "result": result,
            "audio_url": audio_url

        })

    return render(request, "users/UploadForm.html")