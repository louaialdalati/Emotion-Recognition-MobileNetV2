# Real-Time Emotion Recognition using MobileNetV2

Bu proje, Python ve TensorFlow kullanılarak geliştirilmiş gerçek zamanlı bir duygu tanıma sistemidir.

## Özellikler
- **Model:** MobileNetV2 (Transfer Learning)
- **Duygular:** 8 Sınıf (Mutlu, Üzgün, Kızgın, Şaşkın, Korkmuş, İğrenmiş, Aşağılama, Nötr)
- **Performans:** Gerçek zamanlı (Real-time), ~30 FPS
- **Teknolojiler:** TensorFlow, Keras, OpenCV, MediaPipe

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt

2. Sistemi test etmek için (Kamera açılır):
   python test_improved.py

3. Modeli yeniden eğitmek için: Veri setini indirin ve klasöre çıkarın,     ardından:

python train_improved_v2.py