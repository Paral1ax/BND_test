# Тестовый проект для компании BND

## За основу была взята модель Yolo8 medium
> Модель была дообучена на датасете https://universe.roboflow.com/ds/khtn5HOVCb?key=SbVtQb2arn
> Данный датасет состоит из 11833 фотографий
> 1. 9562 Фотографии для обучающей выборки
> 2. 1494 Фотографий на валидационной выборке
> 3. 777 Фотографий на тестовой выборке

## Файл main.py содержит код с предсказанием и отображением детекции на видео

## Файл Yolo8_train_medium_people_count.ipynb содержит код для обучения модели
> Так как для дообучения модели было необходимо GPU с большим количеством памяти, было решено обучать модель в Google Colab
> Поэтому получилось 2 файла с кодом

## Файл best_yolo8m_medium.pt содержит веса дообученной модели

## Файл people_tracking_yolo8small.avi содержит видео с демонстрацией работы Yolo8 small, также обученной на данном датасете

# Демонстрация работы:

### Базовой видео с толпой людей

https://github.com/Paral1ax/BND_test/assets/71229854/5c41b310-256d-47f2-993e-170b3c0e2086

### А так выглядит видео, детектирующее людей:

https://github.com/Paral1ax/BND_test/assets/71229854/54ec2694-a56e-47f0-86d9-8c62790532fe

