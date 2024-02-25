import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.utils import Sequence
from sklearn.feature_extraction.image import extract_patches_2d

class CustomImgGen(Sequence):
    def __init__(self, data_df, batch_size, augmentation_params, df_type):
        self.data_df = data_df
        self.batch_size = batch_size
        self.augmentation_params = augmentation_params
        self.df_type = df_type
        self.datagen = ImageDataGenerator(**self.augmentation_params)
        #self.indexes = np.arange(len(self.data_df))

    def __len__(self):
        return int(np.ceil(len(self.data_df) / self.batch_size))

    def load_image(self, image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_array = np.array(img)
        preprocessed_image = preprocess_input(img_array)
        #img_array = np.stack((img_array,) * 3, axis=-1)
        #scaling_factor = 255.0  # You can adjust this based on your requirement
        #rescaled_image = img_array / scaling_factor
        return preprocessed_image #rescaled_image

    def __getitem__(self, index):
        #start_idx = index * self.batch_size
        #end_idx = min((index + 1) * self.batch_size, len(self.data_df))


        #batch_indexes = self.indexes[start_idx:end_idx]
        batch_indexes = np.random.randint(0, len(self.data_df), size = self.batch_size)
        batch_images = []
        batch_labels = []
        batch_gender = []

        for idx in batch_indexes:
            image_path = self.data_df['path'][idx]
            #label = self.data_df['boneage'][idx]
            label = self.data_df['normalized_boneage'][idx]
            gender = self.data_df['male'][idx]
            image = self.load_image(image_path)
            if self.df_type == 'Training' or self.df_type == 'Validation':
                augmented = self.datagen.random_transform(image)
                #plt.imshow(augmented)
                #plt.show()
                batch_images.append(augmented)
            if self.df_type == 'Test':
                batch_images.append(image)
            batch_labels.append(label)
            batch_gender.append(gender)
        return {'Sex': np.array(batch_gender), 'Images':np.array(batch_images)}, np.array(batch_labels)
    

class CustomImgPatchGen(Sequence):
    def __init__(self, data_df, batch_size, patch_size, max_patches, augmentation_params, df_type):
        self.data_df = data_df
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.augmentation_params = augmentation_params
        self.df_type = df_type
        self.datagen = ImageDataGenerator(**self.augmentation_params)
        self.list_indexes = []
        #self.indexes = np.arange(len(self.data_df))

    def __len__(self):
        return int(np.ceil(len(self.data_df) / self.batch_size))

    def load_image(self, image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_array = np.array(img)
        #preprocessed_image = preprocess_input(img_array)

        #img_array = np.stack((img_array,) * 3, axis=-1)
        #scaling_factor = 255.0  # You can adjust this based on your requirement
        #rescaled_image = img_array / scaling_factor
        return img_array#preprocessed_image #rescaled_image

    def get_indexes(self):
        return np.concatenate(self.list_indexes)

    def __getitem__(self, index):
        #start_idx = index * self.batch_size
        #end_idx = min((index + 1) * self.batch_size, len(self.data_df))


        #batch_indexes = self.indexes[start_idx:end_idx]
        batch_indexes = np.random.randint(0, len(self.data_df)-1, size = self.batch_size)
        #self.list_indexes.append(batch_indexes)
        batch_images = []
        batch_labels = []
        batch_gender = []
        saved_indexes = []

        for idx in batch_indexes:
            image_path = self.data_df['path'][idx]
            #label = self.data_df['boneage'][idx]
            label = self.data_df['normalized_boneage'][idx]
            gender = self.data_df['male'][idx]

            image = self.load_image(image_path)
            patches = extract_patches_2d(image, patch_size = self.patch_size, max_patches = self.max_patches, random_state = 12345)

            for patch in patches:
                if self.df_type == 'Training' or self.df_type == 'Validation':
                    #patch = Image.fromarray(patch, mode='RGB')
                    augmented = self.datagen.random_transform(patch)
                    #plt.imshow(augmented)
                    #plt.show()
                    batch_images.append(augmented)
                if self.df_type == 'Test':
                    batch_images.append(patch)
                batch_labels.append(label)
                batch_gender.append(gender)
                saved_indexes.append(idx)

            #batch_images = np.array(batch_images)
            #batch_labels = np.array(batch_labels)
            #batch_gender = np.array(batch_gender)

            #randomize = np.arange(len(batch_labels))
            #np.random.shuffle(randomize)

            #batch_images = batch_images[randomize]
            #batch_labels = batch_labels[randomize]
            #batch_gender = batch_gender[randomize]
        self.list_indexes.append(saved_indexes)
        return {'Sex': np.array(batch_gender), 'Images':np.array(batch_images)}, np.array(batch_labels)
    

class CustomImgGen_1ch(Sequence):
    def __init__(self, data_df, batch_size, augmentation_params, df_type):
        self.data_df = data_df
        self.batch_size = batch_size
        self.augmentation_params = augmentation_params
        self.df_type = df_type
        self.datagen = ImageDataGenerator(**self.augmentation_params)

    def __len__(self):
        return int(np.ceil(len(self.data_df) / self.batch_size))

    def load_image(self, image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img_array = np.array(img)
        preprocessed_image = preprocess_input(img_array)
        return preprocessed_image

    def __getitem__(self, index):
        batch_indexes = np.random.randint(0, len(self.data_df), size = self.batch_size)
        batch_images = []
        batch_labels = []
        batch_gender = []

        for idx in batch_indexes:
            image_path = self.data_df['path'][idx]
            label = self.data_df['normalized_boneage'][idx]
            gender = self.data_df['male'][idx]
            image = self.load_image(image_path)
            if self.df_type == 'Training' or self.df_type == 'Validation':
                augmented = self.datagen.random_transform(image)
                batch_images.append(augmented[:, :, 0])
            if self.df_type == 'Test':
                batch_images.append(image[:, :, 0])
            batch_labels.append(label)
            batch_gender.append(gender)
        return {'Sex': np.array(batch_gender), 'Images':np.array(batch_images)}, np.array(batch_labels)


img_augmentation_params = {                       
    'samplewise_center': False,
    'samplewise_std_normalization': False,
    'rotation_range': 5,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.01,
    'zoom_range': 0.25,
    'horizontal_flip': True,
    'vertical_flip': False,
    #'brightness_range': (0.8, 1.2),
    'fill_mode': 'reflect',
    #'cval': 0,
    #'data_format': 'channels_last'
}


patch_augmentation_params = {
    'samplewise_center': False,
    'samplewise_std_normalization': False,
    'rotation_range': 5,
    'horizontal_flip': True,
    'vertical_flip': False,
    #'brightness_range': (0.8, 1.2),
    'fill_mode': 'reflect',
}