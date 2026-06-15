# Blueit Predictive Maintenance

Bu proje gelecek olan makine öğrenmesi projelerim için güzel bir base oluşturdu. 
Veri setini seçmemin sebebi şu an üzerinde durduğum bir başka projemin de anomali 
tespiti üzerine olması.

## Veri Seti
[Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

10.000 satır sensör verisinden oluşuyor. Amaç makinenin arızalanıp arızalanmayacağını 
tahmin etmek.

## EDA

Veriyi tanımak için korelasyon matrisi oluşturdum. Target sütunuyla en yüksek 
ilişkisi olan özellikler Torque (0.19) ve Tool Wear (0.11) oldu.Ayrıca sns.countplot kullanarak bir class imbalance fark 
ettim.
## Veri Hazırlığı

İlk başta tüm sayısal özellikleri korudum ama işimize yaramayan UDI ve Product ID 
gibi sütunları sildim, hiçbir katkıları yoktu. Tek katkısı olan string tipindeki 
sütun Type'tı , kullanılan cihazın tipi değiştikçe veriler de değişebileceğinden 
bu özelliği sayıya çevirdim (L→0, M→1, H→2). Sonra veriyi X ve Y olarak ikiye 
ayırdım, Y bizim tahmin etmek istediğimiz Target, X ise geri kalan tüm özellikler. 
Son olarak %80 train %20 test olarak böldük.

Veri setinde ciddi bir sınıf dengesizliği vardı ( 9661 normal, sadece 339 arızalı makine (%96 vs %3)). Bu yüzden modele sadece accuracy'e bakmak yanıltıcı, model her veriye normal diyerek %96 doğruluk alabilirdi. Bunu çözmek için Logistic Regression ve Random Forest'ta class_weight='balanced', XGBoost'ta ise scale_pos_weight=9661/339 parametresi kullandım. Bu sayede model az olan arızalı örneklere daha fazla üstüne düştü.

## Modeller

Bu 3 modeli seçtim çünkü basitten karmaşığa doğru ilerlemeyi istedim.

**Logistic Regression** — > Baz skor almak için kullandım. Train ve Test F1 skoru 0.22-0.23 bandında kaldı, underfitting yapıyordu, model veriyi yeterince öğrenemedi

**Random Forest** — > çoklu ağaç yapısı kullanarak ,Test F1 skoru 0.65'e çıktı, Logistic Regression'a göre çok daha iyi. Ama Train F1 0.99 oldu, overfitting başladı

**XGBoost**  — > son modelim.En dengeli sonucu verdi, Random Foresttan farkı sıralı ağaç yapısı kullnarak hatalarını öğrenebiliyor. Test F1 0.66 ile Random Forest'a yakın ama Recall 0.69'a çıktı  arızaları yakalamak bizim için kritik olduğundan bu fark önemli

## Sonuçlar

| Model | Train F1 | Test F1 |
|---|---|---|
| Logistic Regression | 0.23 | 0.22 |
| Random Forest | 0.99 | 0.65 |
| XGBoost | 0.99 | 0.66 |

Sonuçlara bakınca Logistic Regression underfitting kalıyor, lineer bir model olduğu için scaling gerekiyor. 
Diğer 2 modelimiz ise overfitting yaptı. Ağaç modelleri eğitim verilerini ezberlediği için test error yüksek çıktı

## Kurulum

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Kaynaklar

- [Sentiment Analysis & ML Model Building - Insight Group](https://insight-group.github.io/MFIN7036/sentiment-analysis-machine-learning-build-models.html)
- [Understanding Logistic Regression in Python - DataCamp](https://www.datacamp.com/tutorial/understanding-logistic-regression-python)
- [XGBoost - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/xgboost/)
