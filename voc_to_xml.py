import os
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset to VOC format')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing images and annotations')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory to save VOC format dataset')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset type (train, val, test)')
    return parser.parse_args()

def create_voc_structure(output_dir, dataset):
    """
    Tạo cấu trúc thư mục theo định dạng Pascal VOC    
    """

    dataset_dir = os.path.join(output_dir, dataset)
    images_dir = os.path.join(dataset_dir, 'images')
    annots_dir = os.path.join(dataset_dir, 'annotations')

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annots_dir, exist_ok=True)

    return images_dir, annots_dir

def create_xml_annotation(image_path, objects, output_path):
    """
    Tạo file XML annotation theo định dạng Pascal VOC
    Args:
        image_path: Đường dẫn đến ảnh
        objects: Danh sách các đối tượng trong ảnh
        output_path: Đường dẫn để lưu file XML
    """

    # Lấy thông tin ảnh
    img = Image.open(image_path)
    width, height = img.size

    # Tạo root element
    root = ET.Element('annotation')

    # Thêm thông tin
    folder = ET.SubElement(root, 'folder')
    folder.text = os.path.basename(os.path.dirname(image_path))

    filename = ET.SubElement(root, 'filename')
    filename.text = os.path.basename(os.path.dirname(image_path))

    path = ET.SubElement(root, 'path')
    path.text = image_path

    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # Thêm thông tin kích thước ảnh
    size = ET.SubElement(root, 'size')
    width_elem = ET.SubElement(size, 'width')
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, 'height')
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, 'depth')
    depth_elem.text = '3'  # RGB ảnh có 3 kênh màu

    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0' # 0: không phân đoạn, 1: phân đoạn

    # Thêm thông tin các đối tượng
    for obj in objects:
        object_elem = ET.SubElement(root, 'object')

        name = ET.SubElement(object_elem, 'name')
        name.text = obj['name']

        pose = ET.SubElement(object_elem, 'pose')
        pose.text = 'Unspecified'

        truncated = ET.SubElement(object_elem, 'truncated')
        truncated.text = '0'

        difficult = ET.SubElement(object_elem, 'difficult')
        difficult.text = '0'

        bndbox = ET.SubElement(object_elem, 'bndbox')

        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(obj['bbox'][0])

        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(obj['bbox'][1])

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(obj['bbox'][2])
        
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(obj['bbox'][3]) 

    # Tạo XML tree và lưu file XML
    tree = ET.ElementTree(root)

    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def convert_to_voc(input_dir, output_dir, dataset):
    """
    Chuyển đổi dataset sang định dạng Pascal VOC
    """

    # Tạo cấu trúc thư mục
    images_dir, annots_dir = create_voc_structure(output_dir, dataset)

    # Lấy danh sách các file ảnh 
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Xử lý từng file ảnh
    for img_file in image_files:
        # Đường dẫn file ảnh gốc
        img_path = os.path.join(input_dir, img_file)

        xml_file = os.path.splitext(img_file)[0] + '.xml'
        xml_path = os.path.join(input_dir, xml_file)

        if not os.path.exists(xml_path):
            print(f"Warning: No annotation found for {img_file}, skipping...")
            continue

        # Copy ảnh
        dest_img_path = os.path.join(images_dir, img_file)
        shutil.copy2(img_path, dest_img_path)

        # Copy annot
        dest_xml_path = os.path.join(annots_dir, xml_file)
        shutil.copy2(xml_path, dest_xml_path)

    print(f"Converted {len(image_files)} images to VOC format in {os.path.join(output_dir, dataset)}")

def main():
    args = parse_args

    # Chuyển đổi dataset sang VOC
    convert_to_voc(args.input_dir, args.output_dir, args.dataset)

if __name__ == '__main__':
    main()
