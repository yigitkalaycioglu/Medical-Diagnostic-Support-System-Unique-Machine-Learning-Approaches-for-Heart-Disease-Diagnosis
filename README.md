# Kalp Hastalığı Teşhisinde Standart ve Özgün ML Modelleri

Sakarya Üniversitesi ISE 427 Tıpta Yapay Zeka dersi (2025-2026 Güz) için yaptığım proje. UCI'daki Cleveland kalp hastalığı veri seti üzerinde beş standart sınıflandırıcıyı (lojistik regresyon, SVM, karar ağacı, random forest, KNN) ve her birinin tıbbi veriye göre değiştirdiğim birer versiyonunu karşılaştırdım. Toplam 10 model aynı ön işleme ve aynı test kümesiyle değerlendirildi.

## Veri ve ön işleme

Veri seti 303 hasta kaydı ve 14 sütundan oluşuyor (yaş, cinsiyet, göğüs ağrısı tipi, kan basıncı, kolesterol, maksimum nabız vb. ve hedef: 0 sağlıklı, 1 hasta).

- Eksik değerler medyan/mod ile dolduruldu.
- `chol` ve `trestbps` sütunlarındaki aykırı değerler IQR yöntemiyle baskılandı.
- Kategorik sütunlara one-hot encoding, sayısal sütunlara `StandardScaler` uygulandı.
- Veri sınıf oranı korunarak %80 eğitim, %20 test olarak ayrıldı.

## Özgün versiyonlar

- KNN: Öklid uzaklığı yerine farkların `log(1 + fark²)` toplamını kullanan "log-karesel" mesafe. Büyük farkların etkisini yumuşattığı için aykırı değerlere daha dayanıklı.
- SVM: Laplacian ve Bessel fonksiyonlarını birleştiren hibrit bir çekirdek.
- Lojistik regresyon: Sigmoid yerine ISRU (inverse square root unit) tabanlı bir olasılık fonksiyonu.
- Karar ağacı: Gini/entropi yerine Hellinger uzaklığına dayanan bölme kriteri.
- Random forest: Sabit çoğunluk oyu yerine, test örneğine en yakın komşularda başarılı olan ağaçların oy kullandığı dinamik ağaç seçimi.

Özgün modeller scikit-learn'ün estimator arayüzüyle yazıldı (`BaseEstimator` ve `ClassifierMixin`, SVM için `SVC` alt sınıfı). Bu sayede standart modellerle aynı `fit`/`predict` akışında kullanılabiliyorlar.

## Sonuçlar

Test kümesi: 61 hasta (33 sağlıklı, 28 hasta).

| Model | Doğruluk | Duyarlılık | Özgüllük | F1 |
|---|---|---|---|---|
| KNN (özgün, log-karesel) | 93,4 | 96,4 | 90,9 | 93,1 |
| SVM (özgün, Bessel-Laplacian) | 90,2 | 96,4 | 84,8 | 90,0 |
| Random forest (standart) | 90,2 | 92,9 | 87,9 | 89,7 |
| KNN (standart) | 90,2 | 92,9 | 87,9 | 89,7 |
| Lojistik regresyon (özgün, ISRU) | 88,5 | 92,9 | 84,8 | 88,1 |
| SVM (standart) | 88,5 | 92,9 | 84,8 | 88,1 |
| Lojistik regresyon (standart) | 86,9 | 89,3 | 84,8 | 86,2 |
| Random forest (özgün, dinamik seçim) | 86,9 | 85,7 | 87,9 | 85,7 |
| Karar ağacı (standart) | 73,8 | 75,0 | 72,7 | 72,4 |
| Karar ağacı (özgün, Hellinger) | 54,1 | 7,1 | 93,9 | 12,5 |

En iyi sonucu log-karesel mesafeli KNN verdi: 28 hastanın 27'sini doğru buldu. Teşhiste en önemli metrik hastayı kaçırmamak olduğu için duyarlılığa ayrıca baktım. Özgün KNN ve özgün SVM burada %96,4 ile en yüksek değerde.

Birkaç not:

- Test kümesi küçük olduğu için tek bir hastanın farklı sınıflandırılması doğruluğu yaklaşık 1,6 puan değiştiriyor. Modeller arasındaki küçük farkları bu gözle okumak gerekiyor.
- Hellinger kriterli ağaç bu veri setinde neredeyse herkesi sağlıklı olarak sınıflandırdı (duyarlılık %7). Dinamik seçimli random forest da standart halinin gerisinde kaldı. Yani özgün versiyonların hepsi iyileştirme getirmedi.

## Dosyalar

```
kod dosyaları/
  EDA_Heart_Cleveland_Full.ipynb   keşifsel veri analizi
  Heart_Data_Preprocessing.ipynb   ön işleme, pipeline ve train/test ayrımını kaydediyor
  Heart_Model_Tuning.ipynb         10 modelin eğitimi, ayarlanması ve karşılaştırılması
  Final_Proje_Sonuclari.csv        yukarıdaki sonuç tablosu
proje rapor.docx                   proje raporu
proje sunum.pptx                   sunum
```

## Çalıştırma

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn joblib jupyter
cd "kod dosyaları"
jupyter notebook
```

Notebook'lar dosyaları bulundukları klasörden okuyor. Sırasıyla EDA, ön işleme ve model notebook'u çalıştırılmalı. Ön işleme notebook'u `preprocess_pipeline.pkl` ve `train_test_data.pkl` dosyalarını üretiyor, model notebook'u bunları kullanıyor.

## Veri seti

[UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland alt kümesi). Veri seti kendi lisansına tabidir, repodaki MIT lisansı sadece kod için geçerlidir.
