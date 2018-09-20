## Learning to Color from Language
[Learning to Color from Language, NAACL, 2018 (short)](http://aclweb.org/anthology/N18-2120)


### Instructions : 
1. Download data : https://obj.umiacs.umd.edu/learning_to_color/coco_colors.h5

2. Extract features from black-and-white images
`python extract_image_features.py --input_h5_file coco_colors.h5 --output_h5_file image_features.h5`

3. Then run : 
`python autocolorize_resnet.py --h5_File coco_colors.h5 --features_file ./image_features.h5 --vocab_file_name ./priors/coco_colors_vocab.p --image_save_folder ./trial/ --model_save_file ./models/`

#### Film Activations
![Film Activations](https://raw.githubusercontent.com/superhans/colorfromlanguage/master/images/Activations4.png)
