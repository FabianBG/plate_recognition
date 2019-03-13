import src.ocr_plate as ocr
import sys
import os
from PIL import Image
from src.config import image_chanels_size, alphabet, absolute_max_string_len, steps_per_epoch, epochs, minbatch

if sys.argv[1] == "train":
    m, _ = ocr.model(image_chanels_size, len(alphabet) + 1, absolute_max_string_len) 
    viz_cb = ocr.VizCallback()
    m.fit_generator(
        generator = ocr.batch_generator(minbatch),
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        callbacks = [viz_cb]
    )
elif sys.argv[1] == "plates":
    data, _ = ocr.image_generator(1, img_w=200, img_h=50, downsample_factor=4)
    print('Original:', data['source_str'])
    print('Image:', data['input'][0][:, :, 0].shape)
    image = Image.fromarray(data['input'][0][:, :, 0], 'L')
else:
    m, test = ocr.model(image_chanels_size, len(alphabet) + 1, absolute_max_string_len)
    weight_file = os.path.join("plate_test", os.path.join('weights%s.h5' % sys.argv[1]))
    m.load_weights(weight_file)

    data, _ = ocr.image_generator(1, img_w=200, img_h=50, downsample_factor=4)
    pred = ocr.decode_batch(test, data['input'])
    print('Prediction:', pred)
    print('Original:', data['source_str'])
