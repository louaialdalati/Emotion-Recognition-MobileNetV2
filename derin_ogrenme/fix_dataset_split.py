import os
import shutil
import random
import glob

# --- Configuration ---
# المسار الرئيسي الذي يحتوي على المجلدات الثلاثة (train, test, validation)
# تأكد من أن هذا المسار يطابق المسار الموجود على جهازك
base_dataset_dir = r'D:\projects\yazilim muhendisligi uygulamalari\face_detection\colored face images'

# النسب الجديدة المطلوبة
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# أسماء المجلدات (كما هي في جهازك)
DIR_NAMES = ['train', 'validation', 'test']

# الفئات (المشاعر)
CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def redistribution_images():
    print(f"Starting redistribution in: {base_dataset_dir}")
    
    # التحقق من وجود المجلدات الرئيسية
    for d in DIR_NAMES:
        path = os.path.join(base_dataset_dir, d)
        if not os.path.exists(path):
            print(f"Error: Directory {path} does not exist!")
            return

    # العمل على كل فئة (شعور) على حدة
    for emotion in CLASSES:
        print(f"\nProcessing class: '{emotion}'...")
        
        all_images = []
        
        # 1. تجميع كل الصور من المجلدات الثلاثة
        for d in DIR_NAMES:
            class_path = os.path.join(base_dataset_dir, d, emotion)
            if os.path.exists(class_path):
                # البحث عن الصور بامتدادات مختلفة
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images.extend(glob.glob(os.path.join(class_path, ext)))
                all_images.extend(images)
            else:
                # إنشاء المجلد إذا لم يكن موجوداً لتجنب الأخطاء لاحقاً
                os.makedirs(class_path, exist_ok=True)

        total_images = len(all_images)
        print(f"  - Found total {total_images} images for '{emotion}'.")
        
        if total_images == 0:
            print(f"  - Warning: No images found for {emotion}. Skipping.")
            continue

        # 2. خلط الصور عشوائياً
        random.shuffle(all_images)
        
        # 3. حساب نقاط التقسيم
        train_end = int(total_images * TRAIN_RATIO)
        val_end = train_end + int(total_images * VAL_RATIO)
        
        train_files = all_images[:train_end]
        val_files = all_images[train_end:val_end]
        test_files = all_images[val_end:]
        
        print(f"  - New Split -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # 4. نقل الملفات (دالة مساعدة للنقل)
        def move_files(files, target_split_name):
            target_dir = os.path.join(base_dataset_dir, target_split_name, emotion)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            for file_path in files:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(target_dir, file_name)
                
                # نقل الملف فقط إذا لم يكن في مكانه الصحيح بالفعل
                if file_path != dest_path:
                    try:
                        shutil.move(file_path, dest_path)
                    except Exception as e:
                        print(f"    Error moving {file_name}: {e}")

        # تنفيذ النقل
        move_files(train_files, 'train')
        move_files(val_files, 'validation')
        move_files(test_files, 'test')
        
    print("\nRedistribution Complete! Your dataset is now balanced.")

if __name__ == "__main__":
    # تأكيد من المستخدم قبل البدء
    print("This script will move files between train/validation/test folders.")
    print("Please make sure you have a backup if the data is critical.")
    confirm = input("Type 'yes' to continue: ")
    
    if confirm.lower() == 'yes':
        redistribution_images()
    else:
        print("Operation cancelled.")