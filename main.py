import src.ocr_plate as ocr
import sys
import os
import datetime
from exporter import export_h5_to_pb
from PIL import Image
from src.config import image_chanels_size, alphabet, absolute_max_string_len, steps_per_epoch, epochs, minbatch, image_size

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
elif sys.argv[1] == "samples":
    for _ in range(0, int(sys.argv[2])):
        ocr.get_plate(image_size, False, False, './')
elif sys.argv[1] == "export":
    m, test = ocr.model(image_chanels_size, len(alphabet) + 1, absolute_max_string_len)
    weight_file = os.path.join("plate_test", os.path.join('weights%s.h5' % sys.argv[2]))
    m.load_weights(weight_file)
    model_name = './' + datetime.datetime.now().strftime("%Y-%m-%d%H:%M:%S") + '.h5'
    m.save(model_name)
    export_h5_to_pb(model_name, model_name.replace('.h5', '.pb'), {'ctc': ocr.ctc})
else:
    m, test = ocr.model(image_chanels_size, len(alphabet) + 1, absolute_max_string_len)
    weight_file = os.path.join("plate_test", os.path.join('weights%s.h5' % sys.argv[1]))
    m.load_weights(weight_file)
    data, _ = ocr.image_generator(1, img_w=200, img_h=50, downsample_factor=4, show_image=True)
    pred = ocr.decode_batch(test, data['input'])
    print('Prediction:', pred)
    print('Original:', data['source_str'])
