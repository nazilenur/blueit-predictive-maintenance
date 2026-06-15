# Blueit Predictive Maintenance

Bu proje gelecek olan makine öğrenmesi projelerim için güzel bir base oluşturdu. 
Veri setini seçmemin sebebi şu an üzerinde durduğum bir başka projemin de anomali 
tespiti üzerine olması.

## Veri Seti
[Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

10.000 satır sensör verisinden oluşuyor. Amaç makinenin arızalanıp arızalanmayacağını 
tahmin etmek.

## Veri Hazırlığı

İlk başta tüm sayısal özellikleri korudum ama işimize yaramayan UDI ve Product ID 
gibi sütunları sildim, hiçbir katkıları yoktu. Tek katkısı olan string tipindeki 
sütun Type'tı — kullanılan cihazın tipi değiştikçe veriler de değişebileceğinden 
bu özelliği sayıya çevirdim (L→0, M→1, H→2). Sonra veriyi X ve y olarak ikiye 
ayırdım, y bizim tahmin etmek istediğimiz Target, X ise geri kalan tüm özellikler. 
Son olarak %80 train %20 test olarak böldük.

## Modeller

Bu 3 modeli seçtim çünkü basitten karmaşığa doğru ilerlemeyi istedim.

**Logistic Regression** — temel değerleri görmek için kullandım, underfitting oldu, 
yeterli değildi.

**Random Forest** — çoklu ağaç yapısı kullanarak çok daha doğru sonuç verdi ve 
tutarlıydı.

**XGBoost** — son modelim. Random Forest'tan farkı ağaç yapılarını sıralı 
çalıştırarak sürekli kendini düzeltmesi. Endüstride çok yaygın kullanılıyor.

## Sonuçlar

| Model | Train F1 | Test F1 |
|---|---|---|
| Logistic Regression | 0.23 | 0.22 |
| Random Forest | 0.99 | 0.65 |
| XGBoost | 0.99 | 0.66 |

Sonuçlara bakınca Logistic Regression underfitting kalıyor, lineer bir model olduğu için scaling gerekiyor. 
Diğer 2 modelimiz ise overfitting yapıyor ağaç modelleri eğitim verilerini ezberledi bu yüzden test error yüksek çıktı. Bir sonraki adımda hyperparameter tuning ile çözülebilir

## Kurulum

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
