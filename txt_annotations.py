import os

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

classes_path    = 'model_data/cls_classes.txt'
datasets_path   = '/public/home/tongshq/EEG-dataset-1'

sets            = ["train", "test"]
classes, _      = get_classes(classes_path)

if __name__ == "__main__":
    for se in sets:
        list_file = open('cls_' + se + '.txt', 'w')

        datasets_path_t = os.path.join(datasets_path, se)
        types_name      = os.listdir(datasets_path_t)
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id = classes.index(type_name)
            
            photos_path = os.path.join(datasets_path_t, type_name)
            photos_name = os.listdir(photos_path)
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.csv']:
                    continue
                list_file.write('%s'%(os.path.join(photos_path, photo_name)) + '\t' + str(cls_id))
                list_file.write('\n')
        list_file.close()

