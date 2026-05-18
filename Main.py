import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn.model_selection import LeaveOneOut

# 1. Load Dataset
def load_data(n_samples=50):
    print(f"--- Membaca dataset ({n_samples} sampel) ---")
    try:
        path = "emnist-balanced-train.csv" 
        data = pd.read_csv(path, header=None)
        data_subset = data.head(n_samples)
        
        y = data_subset.iloc[:, 0].values
        X_raw = data_subset.iloc[:, 1:].values
        
        X_processed = []
        for img_array in X_raw:
            img = img_array.reshape(28, 28).T # Transpose agar tegak
            X_processed.append(img)
            
        return np.array(X_processed), y
    except FileNotFoundError:
        print("Error: File CSV tidak ditemukan.")
        return None, None

# 2. Ekstraksi Fitur HOG
def extract_hog_features(images):
    print("--- Mengekstraksi Fitur HOG ---")
    hog_features = []
    for img in images:
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)

# 3. Visualisasi Hasil (Fungsi Baru)
def plot_results(y_true, y_pred, X_img):
    # Plot 1: Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - EMNIST Classification')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Plot 2: Menampilkan beberapa contoh gambar dataset
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(X_img[i], cmap='gray')
        plt.title(f"Label: {y_true[i]}")
        plt.axis('off')
    
    print("\n--- Menampilkan Grafik Visualisasi... ---")
    plt.show()

# 4. Evaluasi LOOCV
def train_and_evaluate(X_feat, y, X_img):
    print("--- Menjalankan LOOCV ---")
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    
    for i, (train_index, test_index) in enumerate(loo.split(X_feat)):
        X_train, X_test = X_feat[train_index], X_feat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(X_train, y_train)
        
        y_pred.append(clf.predict(X_test)[0])
        y_true.append(y_test[0])
        print(f"Proses: {i+1}/{len(X_feat)}", end='\r')
    
    # Cetak hasil di terminal
    print("\n\n--- HASIL EVALUASI ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    
    # Panggil fungsi visualisasi
    plot_results(y_true, y_pred, X_img)

# --- EXECUTION ---
if __name__ == "__main__":
    n_data = 50 # Anda bisa naikkan ke 100 jika spek laptop mumpuni
    X_img, y_label = load_data(n_samples=n_data)
    
    if X_img is not None:
        features = extract_hog_features(X_img)
        train_and_evaluate(features, y_label, X_img)