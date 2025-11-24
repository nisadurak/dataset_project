# 🎮 InfiniteX — Kamera Perspektifi Sınıflandırma

> Oyun ekran görüntülerini analiz ederek kameranın **First-Person**, **Third-Person**, **Isometric**, **Top-Down** veya **Side-Scroller** perspektiflerinden hangisine ait olduğunu tahmin eden yapay zeka projesi.


## İçerik Videsu

[▶️ Videoyu İzle](https://github.com/user-attachments/assets/885a8cd0-a84f-4caa-b8ff-e117c951268a)
---

## 🚀 Özellikler

- 🔍 Görsel sınıflandırma: 5 perspektif türü  
- ⚙️ İki model karşılaştırması: `ResNet50` vs `GameCamNet (Custom CNN)`  
- 💻 Web arayüzü: Flask + HTML + CSS (InfiniteX Neon Tasarımı)  
- 🧠 Eğitim ortamı: PyTorch  


---

## 🧩 Kullanılan Modeller

### **1. GameCamNet (Custom CNN)**
- Veri seti her kategori için belirli oyunların videolarından görüntü alınarak toplanmıştır. 
- Sıfırdan tasarlanmış, hafif mimari
- 4 adet Conv–BatchNorm–ReLU bloğu  
- 2 Fully Connected katman  
- Validation doğruluğu: **%86**

### **2. ResNet50 (Transfer Learning)**
- ImageNet üzerinde önceden eğitilmiş  
- Son katman 5 sınıfa göre yeniden eğitildi  
- Validation doğruluğu: **%99**

---

## ⚙️ Kurulum

```bash
# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Gereksinimleri yükle
pip install -r requirements.txt

# Uygulamayı başlat
python app.py

```

# Uygulama çalıştığında tarayıcıdan şu adrese git:
👉 http://127.0.0.1:5000

```bash
Kullanıcı görüntü yükler →
   Flask dosyayı kaydeder →
      Görüntü ön işleme (resize + normalize) →
         ResNet50 ve CNN ile tahmin →
             Softmax olasılık hesaplama →
                 Top-3 sonuçlar & açıklama →
                     Arayüzde gösterim

```

| Özellik           | GameCamNet (CNN)               | ResNet50             |
| ----------------- | ------------------------------ | -------------------- |
| Doğruluk          | ~%86                           | ~%99                 |
| Eğitim Süresi     | 25 dk                          | 1.5 saat             |
| Boyut             | 11 MB                          | 97 MB                |
| Güçlü Olduğu Alan | Basit sahneler                 | Gerçekçi 3D sahneler |
| Zayıf Nokta       | Benzer perspektiflerde karışma | Büyük model boyutu   |


| Perspektif           | Görsel                                                     | ResNet50 Doğruluğu |
| -------------------- | ---------------------------------------------------------- | ------------------ |
| **Isometric**        | ![Isometric](static/uploads/isometric-analiz.png)          | %97.2              |
| **Top-Down**         | ![Top-Down](static/uploads/top-down-analiz.png)            | %99.8              |
| **Third-Person**     | ![Third-Person](static/uploads/third-analiz.png)           | %99.5              |
| **Side-Scroller**    | ![Side-Scroller](static/uploads/sidescroll-analiz.png)     | %83.1              |
| **First-Person (1)** | ![First-Person](static/uploads/first-person-analiz.png)    | %70.4              |
| **First-Person (2)** | ![First-Person 2](static/uploads/first-person-analiz2.png) | %84.8              |


InfiniteX/
│
├── app.py                # Flask sunucusu
├── models/
│   ├── resnet_model.pth
│   └── cnn_model.pth
├── static/
│   ├── style.css
│   └── uploads/          # Yüklenen görseller
├── templates/
│   ├── index.html
│   ├── result.html
│   └── docs.html
└── README.md


👩‍💻 Geliştirici Notu

Bu proje, oyun içi kamera türlerini makine öğrenmesiyle otomatik olarak ayırt etmeyi amaçlayan bir staj projesidir.
Model, hem düşük donanımda çalışabilmesi hem de yüksek doğrulukla sonuç üretebilmesi için optimize edilmiştir.


---

