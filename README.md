# 📌 LSTM Tabanlı Yelp Yorum Puanı Tahmin Modeli
## Metin yorumlarından 1–5 arası puan tahmini yapan LSTM tabanlı bir regresyon modeli.

##  Proje Özeti

Bu proje, Yelp Review Full veri setini kullanarak kullanıcı yorumlarının sayısal skorunu tahmin etmektedir.
Metinler tokenize edilip dizilere dönüştürülmüş, ardından LSTM katmanları ile eğitilen bir derin öğrenme modeli oluşturulmuştur. Amaç; verilen bir metin yorumundan, kullanıcının kaç puan vermiş olabileceğini öngörmektir.

## 🧠 Kullanılan Yöntemler & Teknolojiler
* Python, TensorFlow, Keras, scikit-learn
* NLP Pipeline: Tokenizer, Sequence Padding
* LSTM tabanlı derin öğrenme modeli
* Regresyon (MSE Loss, MAE Metric)
* MinMaxScaler ile etiket ölçekleme
* Model kaydetme & yükleme (H5 formatı)
