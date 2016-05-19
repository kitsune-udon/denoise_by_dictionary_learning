import numpy as np
import sys,os,json,imageio,omp

def denoise(out_path, in_path, patch_size, D, sparsity):
    src_img = imageio.read_image_L(in_path)
    dst_img = np.zeros((src_img.shape))
    h, w = src_img.shape
    pw, ph = patch_size
    xn, yn = w/pw, h/ph
    for j in xrange(yn):
        for i in xrange(xn):
            print "denoise ({}/{}) : {}".format(j*xn+i+1, xn*yn, in_path)
            x, y = i * pw, j * ph
            v0 = src_img[y:y+ph, x:x+pw].ravel()
            c = omp.omp(v0, D, sparsity)
            v1 = np.dot(D, c)
            dst_img[y:y+ph, x:x+pw] = v1.reshape((ph, pw))
    imageio.write_image_L(out_path, dst_img)

def filename_of_out_image(in_filename):
    fn = os.path.basename(in_filename)
    return fn+".denoised.png"

sys.argv.pop(0)
setting_path = sys.argv.pop(0)
params = json.load(open(setting_path))
working_dir = os.path.join(params['working_dir'], params['setting_id'])
dict_path = os.path.join(working_dir, params['dictionary_filename'])
dictionary = np.load(dict_path)
patch_size = (params['pw'], params['ph'])
sparsity = params['sparsity']

if len(sys.argv) > 0:
    noise_images = [params['noise_images'][int(sys.argv[0])]]
else:
    noise_images = params['noise_images']

for target_path in noise_images:
    out_path = os.path.join(working_dir, filename_of_out_image(target_path))
    denoise(out_path, target_path, patch_size, dictionary, sparsity)
