import numpy as np
import scipy
import scipy.sparse.linalg as spla
from PIL import Image
import os
import time
import argparse
from log_utils import log
import cv2


parser = argparse.ArgumentParser()

parser.add_argument('--input_path',
    type=str, required=True, help='Which solid state lidar data is used')
parser.add_argument('--guidance_path',
    type=str, required=True, help='guidance image data')
parser.add_argument('--output_path',
    type=str, required=True, help='Output path for saving the prediction results')
parser.add_argument('--eval_output_path',
    type=str, required=True, help='Output path for evaluation metric')
parser.add_argument('--use_gpu',
    type=str, required=True, choices=["True", "False"], help='use GPU or not')
parser.add_argument('--winRad',
    type=int, required=True, help='The radius size of the neighbours')
parser.add_argument('--kernel',
    type=str, required=True, choices=["cross", "full"], help='The shape of the neighbours')

args = parser.parse_args()

sparse_path = args.input_path
guidance_path = args.guidance_path
output_folder = args.output_path
eval_output_path = args.eval_output_path
log_path = os.path.join(eval_output_path, 'results.txt')
if args.use_gpu == "True":
    use_gpu = True
else:
    use_gpu = False
winRad = args.winRad
kernel = args.kernel

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


sparse_list = get_imlist(sparse_path)
guidance_list = get_imlist(guidance_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def compute_gradients(image):
    # Check if the image has multiple channels
    if len(image.shape) > 2 and image.shape[2] > 1:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # If the image is already single-channel, continue without conversion
        gray_image = image.astype('float32')

    # # Compute gradients using Sobel operator
    # dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    # dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    # gradient_magnitude = np.sqrt(dx**2 + dy**2)

    # Compute gradients using Scharr operator
    dx = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
    dy = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    # # Compute gradients using Laplace filter
    # gradient_magnitude = cv2.Laplacian(gray_image, cv2.CV_32F)
    # gradient_magnitude = np.abs(gradient_magnitude)
    
    return gradient_magnitude

def compute_weights(image, kernel):
    if len(image.shape) > 2 and image.shape[2] > 1:
        height, width, _ = image.shape
    else:
        # If the image is single-channel
        height, width = image.shape
    
    if kernel == "cross":
        weights = np.full((height, width, 4), np.nan)  # Initialize with NaN values
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]  # Define relative positions of neighboring pixels [upper(dy,dx),left,right,lower]
    # elif kernel == "full":
    #     weights = np.full((height, width, 8), np.nan)  # Initialize with NaN values
    #     neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # Define relative positions of neighboring pixels
    # Compute gradients
    gradient_magnitude = compute_gradients(image)
    
    # Compute weights
    for y in range(height):
        for x in range(width):
            for i, (dy, dx) in enumerate(neighbors):
                ny, nx = y + dy, x + dx
                # Check if neighbor is within image bounds
                if 0 <= ny < height and 0 <= nx < width:
                    # print(gradient_magnitude[ny, nx])
                    offset = 1 
                    weights[y, x, i] = 1 / (offset + gradient_magnitude[ny, nx]) # inverse proportional to gradient        

            # Normalize weights
            valid_indices = ~np.isnan(weights[y, x])
            total_weight = np.sum(weights[y, x][valid_indices])
            if total_weight != 0:
                weights[y, x][valid_indices] /= total_weight
            
    # Add a fifth channel with all values of 1
    ones_channel = np.ones_like(weights[:, :, :1])  # Create a channel of ones with the same shape as weights

    # Concatenate the new channel with the weights
    final_weights = np.concatenate((weights, ones_channel), axis=2)
    
    # Flatten the array along the columns first
    flattened_weights = final_weights.transpose(1, 0, 2).flatten()
    
    # Filter out NaN elements
    valid_indices = ~np.isnan(flattened_weights)
    flattened_weights = flattened_weights[valid_indices]
    
    # Multiply non-zero elements by -1 except when the value is 1
    non_one_indices = flattened_weights != 1
    flattened_weights[non_one_indices] *= -1
    
    return flattened_weights
    

def colorization(imgDepth, guidance, kernel='cross', winRad=1): # kernel= 'cross' or 'full', 'cross uses 4 nearest neighbours, 'full' uses 8 nearest neighbours, default window radius = 1
    imgNotValid = imgDepth == 0 # if pixel value is 0, then it is True, else False
    maxImgAbsDepth = np.max(imgDepth)
    imgDepth[imgDepth > maxImgAbsDepth] = maxImgAbsDepth
    # imgDepth = imgDepthInput / maxImgAbsDepth
    # imgDepth[imgDepth > 1] = 1
    H, W = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).T # [[0 375 750 ... 467625 468000 468375][1 376 751 ... 467626 468001 468376]...[374 749 1124 ... 467999 468374 468749]]
    knownValMask = (~imgNotValid).astype(int) # 0 when the pixel value is 0 (True), 1 when the pixel value is not 0 (False)
    # Initialize variables
    len_ = 0
    absImgNdx = 0
    # the windows determine how far the neighbouring pixels affecting the center pixel
    if kernel=='cross':
        len_window = (winRad * 4) + 1  # cross kernel
    elif kernel=='full':
        len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    # Original loop
    # array with length numPix * len_window, 468750 *len_window
    cols = np.zeros(len_zeros, dtype=int)
    rows = np.zeros(len_zeros, dtype=int)
    vals = np.zeros(len_zeros, dtype=float)

    # loop for all pixels as the center pixel i,j
    for j in range(W):
        for i in range(H):
            nWin = 0
            # loop for all neighbouring pixels affecting the center pixel depending on your window radius
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    # if it is the center pixel, continue to next jj
                    if ii == i and jj == j:
                        continue
                    if kernel == 'cross':
                        if abs(ii - i) > 0 and abs(jj - j) > 0:
                            continue

                    rows[len_] = absImgNdx # store index from 0 to H*W, e.g. H = 375,W=1250, index is from 0 to 468750, [0,0,1,1,1,,2,2,2,...,468750,468750]
                    cols[len_] = indsM[ii, jj] # store the value at [ii,jj], values are indices, indices from 0 to H*W, e.g. H = 375,W=1250, index is from 0 to 468750, [375 1 376 , 0 375 376 ,2,377,3,...]
                    # print([ii,jj])
                    # indsM
                    # [[     0    375    750 ... 467625 468000 468375]
                    #  [     1    376    751 ... 467626 468001 468376]
                    #  [     2    377    752 ... 467627 468002 468377]
                    #  ...
                    #  [   372    747   1122 ... 467997 468372 468747]
                    #  [   373    748   1123 ... 467998 468373 468748]
                    #  [   374    749   1124 ... 467999 468374 468749]]

                    # For row, first pixel index 0, it has 2 neighbours,second pixel index 1 has 3 neighbours, third pixel 2 has 3 neighbours
                    # rows[len_] = absImgNdx # store index from 0 to H*W, e.g. H = 375,W=1250, index is from 0 to 468750, [0,0,1,1,1,,2,2,2,...,468750,468750]
                    # For column index 0 has neighbours 375,1, index 1 has neighbours 0,375,376,2,377
                    # cols[len_] = indsM[ii, jj] # store the value at [ii,jj], values are indices, indices from 0 to H*W, e.g. H = 375,W=1250, index is from 0 to 468750, [375 1 , 0 376 2 ,...]
                    # gval[] for the first pixel is [ 0.  0.  0.  0. -1. -1. -1. -1. -1.]

                    len_ = len_ + 1
                    nWin = nWin + 1

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx # index for the center pixel , len_ = 3,9,15,...
            cols[len_] = absImgNdx 

            len_ += 1
            absImgNdx += 1


    rows = rows[:len_] # rows = [0.     0.    0.   1.   1.     1.   1.   2.    2.      2.      2.   3. 3. 3. 3.  ...], for pixel 0, it has 2 neighbours and itself
    cols = cols[:len_] # cols = [375.   1.    0.   0.   376.   2.   1.   1.    377.    3.      2.                           ...]
    vals = guidance 
    
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix)) # should be a large sparse matrix with diagonals are the blocks containing the neighbouring pixels

    rows = np.arange(0, numPix) # rows = [0      1     2     3    4    5    6     7    8     9    10    11   12    13                           ...]
    cols = np.arange(0, numPix) # cols = [0      1     2     3    4    5    6     7    8     9    10    11   12    13                           ...]
    vals = knownValMask.T.reshape(numPix) # values are mostly zero on the pixels that no sparse point cloud and 1 on valid pixels
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A += G
    b = sparse.flatten('F')

    new_vals = spla.spsolve(A, b)
    output = np.reshape(new_vals, (H, W), 'F')

    return output

# create matrix using gpu, @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @nb.njit
if use_gpu == True:
    import numba as nb
    from numba import njit

    @nb.njit
    def matrix_gpu(imgDepth, guidance, kernel='cross', winRad=1): # kernel= 'cross' or 'full', 'cross uses 4 nearest neighbours, 'full' uses 8 nearest neighbours,default window radius = 1
        H, W = imgDepth.shape
        numPix = H * W
        indsM = np.arange(numPix).reshape((W, H)).T # [[0 375 750 ... 467625 468000 468375][1 376 751 ... 467626 468001 468376]...[374 749 1124 ... 467999 468374 468749]]
        # Initialize variables
        len_ = 0
        absImgNdx = 0
        # the windows determine how far the neighbouring pixels affecting the center pixel
        if kernel=='cross':
            len_window = (winRad * 4) + 1  # cross kernel
        elif kernel=='full':
            len_window = (2 * winRad + 1) ** 2
        len_zeros = numPix * len_window

        # Original loop
        # array with length numPix * len_window, 468750 *len_window, with values = -1
        cols = np.zeros(len_zeros, dtype=np.int32)  # Specify dtype as np.int64
        rows = np.zeros(len_zeros, dtype=np.int32)

        # loop for all pixels as the center pixel i,j
        for j in range(W):
            for i in range(H):
                nWin = 0
                # loop for all neighbouring pixels affecting the center pixel depending on your window radius
                for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                    for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                        # if it is the center pixel, continue to next jj
                        if ii == i and jj == j:
                            continue
                        if kernel == 'cross':
                            if abs(ii - i) > 0 and abs(jj - j) > 0:
                                continue

                        rows[len_] = absImgNdx # store index from 0 to H*W, e.g. H = 375,W=1250, index is from 0 to 468750, [0,0,1,1,1,,2,2,2,...,468750,468750]
                        cols[len_] = indsM[ii, jj] # store the value at [ii,jj], values are indices, indices from 0 to H*W, e.g. H = 375,W=1250, index is from 0 to 468750, [375 1 376 , 0 375 376 ,2,377,3,...]
                        # print([ii,jj])
                        # indsM
                        # [[     0    375    750 ... 467625 468000 468375]
                        #  [     1    376    751 ... 467626 468001 468376]
                        #  [     2    377    752 ... 467627 468002 468377]
                        #  ...
                        #  [   372    747   1122 ... 467997 468372 468747]
                        #  [   373    748   1123 ... 467998 468373 468748]
                        #  [   374    749   1124 ... 467999 468374 468749]]

                        # For row, first pixel index 0, it has 2 neighbours,second pixel index 1 has 3 neighbours, third pixel 2 has 3 neighbours
                        # rows[len_] = absImgNdx # store index from 0 to H*W, e.g. H = 375,W=1250, index is from 0 to 468750, [0,0,1,1,1,,2,2,2,...,468750,468750]
                        # For column index 0 has neighbours 375,1, index 1 has neighbours 0,375,376,2,377
                        # cols[len_] = indsM[ii, jj] # store the value at [ii,jj], values are indices, indices from 0 to H*W, e.g. H = 375,W=1250, index is from 0 to 468750, [375 1 , 0 376 2 ,...]
                        # gval[] for the first pixel is [ 0.  0.  0.  0. -1. -1. -1. -1. -1.]

                        len_ = len_ + 1
                        nWin = nWin + 1

                # # Now the self-reference (along the diagonal).
                rows[len_] = absImgNdx # index for the center pixel , len_ = 3,9,15,...
                cols[len_] = absImgNdx 

                len_ += 1
                absImgNdx += 1
        

        rows_A = rows[:len_] # rows = [0.     0.    0.   1.   1.     1.   1.   2.    2.      2.      2.   3. 3. 3. 3.  ...], for pixel 0, it has 2 neighbours and itself
        cols_A = cols[:len_] # cols = [375.   1.    0.   0.   376.   2.   1.   1.    377.    3.      2.                           ...]
        vals_A = guidance

        return rows_A, cols_A ,vals_A

    
def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array
    depth = np.array(Image.open(filename), dtype=int)

    return depth

num_sample = len(sparse_list)
time_elapse = 0

for i in range(len(sparse_list)):
    output_name = os.path.basename(sparse_list[i]).split('/')[-1]
    output_path = os.path.join(output_folder, output_name)
    sparse = depth_read(sparse_list[i])
    image_guidance = depth_read(guidance_list[i])

    guidance_weight = compute_weights(image_guidance, kernel)

    if len(sparse.shape) == 3:
        sparse = sparse[:, :, 0]
    
    start_time = time.time()

    if use_gpu == True:
        start_time = time.time()
        imgNotValid = sparse == 0
        knownValMask = (~imgNotValid).astype(int)
        H, W = sparse.shape
        numPix = H * W

        # -----------------------------------------------------------------------------------#
        # # use numba to create matrix and scipy to solve the equation
        rows_A, cols_A ,vals_A = matrix_gpu(sparse, guidance_weight, kernel=kernel, winRad=winRad)
        rows_G = np.arange(0, numPix)
        cols_G = np.arange(0, numPix)
        vals_G = knownValMask.T.reshape(numPix)

        A = scipy.sparse.csr_matrix((vals_A, (rows_A, cols_A)), (numPix, numPix))  
        G = scipy.sparse.csr_matrix((vals_G, (rows_G, cols_G)), (numPix, numPix))
        A += G
        b = sparse.flatten('F')

        # Solve the sparse linear system Ax=b, the default sparse solver is UMFPACK when available
        # new_vals = spla.spsolve(A, b)
        # if the matrix is sysmmetric or near-symmetric
        new_vals = spla.spsolve(A, b, permc_spec="MMD_AT_PLUS_A") 

        result = np.reshape(new_vals, (H, W), 'F')

    else:
        result = colorization(sparse, guidance_weight, kernel=kernel, winRad=winRad)

    time_elapse +=  time.time() - start_time
    print("--- %s seconds ---" % (time.time() - start_time))

    if np.amax(result) < 256:
        result = np.round((result / 255) * 65535).astype(np.uint16)
    result = np.uint32(result)
    result = Image.fromarray(result, mode='I') 
    result.save(output_path) # result is 0 to 65535

time_elapse_per_sample = time_elapse/float(num_sample)
log('Total testing time: {:.2f} min  Average testing time per sample: {:.2f} s'.format(
    time_elapse / 60.0, time_elapse_per_sample ),
    log_path)