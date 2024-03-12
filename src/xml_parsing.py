import xml.etree.ElementTree as ET
from PIL import Image
import os

# XML файл
xml_file = r'C:\Users\acer\Desktop\annotations.xml'
# Папка с изображениями
image_folder = r'C:\Users\acer\Desktop\ml\\'
# Папка для сохранения 
output_folder = r'C:\Users\acer\Desktop\objects\\'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Парсим XML файл
tree = ET.parse(xml_file)
root = tree.getroot()

for image in root.findall('image'):
    image_name = image.attrib['name']
    try:
        img = Image.open(image_folder + image_name)
        print(f"открыто изображение: {image_name}")
    except Exception as e:
        print(f"ошибка при открытии: {image_name}: {e}")

    for obj in image.findall('object'):
        try:
            obj_name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)
            print("Обработка объекта...")
        except Exception as e:
            print(f"ошибка: {e}")

        # Вырезаем и сохраняем в папку
        obj_img = img.crop((xmin, ymin, xmax, ymax))
        obj_img.save(f"{output_folder}{obj_name}_{image_name}")
        print(f"сохранены: {output_folder}{obj_name}_{image_name}")

print("конец")
