# Implementation Decision Tree

## Amaç

Önce şu soruyu temizlemek:

> ECT 2-step gerçekten zayıf mı, yoksa bizim mevcut setup bunu yanlış mı gösteriyor?

Bu netleşmeden ana implementation fikrini kilitlememek daha doğru.

## Önce yapılacak 3 doğrulama

### 1. `mid_t` sweep

Sabit `mid_t=0.821` yerine birkaç ara zaman deneriz.

Hedef:

- 2-step ciddi şekilde iyileşiyor mu?
- yoksa her `mid_t`'de fark küçük mü kalıyor?

### 2. Official checkpoint sanity check

Authors'ın yayınladığı checkpoint varsa, aynı eval pipeline ile onu ölçeriz.

Hedef:

- official modelde 2-step iyi çıkıyorsa, bizim pipeline büyük ihtimalle doğru
- official modelde de 2-step kötü çıkıyorsa, önce eval varsayımlarını sorgularız

### 3. Kısa continuation

1980-kimg checkpoint'tan biraz daha devam ederiz.

Hedef:

- ek tuning ile 1-step ve 2-step ayrışıyor mu?
- ayrışıyorsa problem daha çok kısa-budget rejiminde

## Karar ağacı

### Durum A

`mid_t` sweep veya continuation güçlü 2-step kazancı verirse:

- ana hikâye artık "dead second step" olmaz
- ana extension yönü schedule-aware veya trajectory-aware bir yöntem olur

En mantıklı yön:

- segment / trajectory conditioning
- prefix handoff
- schedule-sensitive few-step design

### Durum B

Official checkpoint iyi, bizim checkpoint kötü ise:

- problem büyük ihtimalle ECT'nin kendisinde değil
- problem bizim kısa tuning budget veya local rejimde

En mantıklı yön:

- short-budget ECT failure üzerine yeni yöntem
- düşük bütçede useful second step üretmeye çalışan yöntem

### Durum C

Hiçbir check 2-step'i kurtarmazsa:

- mevcut rejimde ikinci step gerçekten etkisiz
- bu durumda ana implementation fikri `OAR-ECT` olur

## OAR-ECT ne zaman ana fikir olur

Şu koşullarda:

- best `mid_t` ile bile 2-step kazancı küçük kalır
- continuation ile bile anlamlı fark açılmaz
- official sanity check bizim pipeline'ı çürütmez

O zaman ana hipotez şu olur:

> kısa-budget ECT, useful few-step generator değil, güçlü bir one-shot projector gibi davranıyor

Ve çözüm:

> ikinci step'i aynı ağla tekrar etmek yerine, ECT'nin coarse output'u üzerinde eğitilen küçük bir asymmetric refiner eklemek

## Şu anki en doğru pozisyon

Bugün için en dürüst cümle şu:

> anomaly gerçek görünüyor, ama henüz yapısal mı yoksa setup kaynaklı mı kesin söylemek için erken

Yani:

- analysis var
- şüphecilik var
- extension adayı var
- ama final kilit için 3 doğrulama daha lazım

## Dosyalar

Bu süreç için ana çalışma notebook'u:

- [ect_validation_extension_workbench.ipynb](/Users/batuhankaraman/Downloads/COMP447/project/ect_validation_extension_workbench.ipynb)

Ana extension adayı notu:

- [FINAL_IMPLEMENTATION_IDEA.md](/Users/batuhankaraman/Downloads/COMP447/project/FINAL_IMPLEMENTATION_IDEA.md)
