# Purpose: 
#     remove bright pixels caused by energetic particles for IRIS raster image. The pixels with spikes are replaced
#     with nearby pixels at the same wavelength.
# Input:
#     im: input image. 2-D image with [distance, wavelength], a general map with [ny, nx] is allowed.
#     int_lim: if intensities are larger than this value (intensity limit), these pixels will be candidate with spikes.
#     n_isolate: if the number of connected pixels is less than that value, they will be treated as spikes.
# Output:
#     im2: output image.
# Possible improvement:
#     use local median value to replace int_lim.
def iris_despike(im, int_lim=30, n_isolate=9):
    import skimage.morphology as sm
    from skimage.measure import label
    from scipy.ndimage import median_filter
    ind = np.nonzero(im>-100)
    iy0 = np.min(ind[0])
    iyt = np.max(ind[0])
    ix0 = np.min(ind[1])
    ixt = np.max(ind[1])
    imc = im[iy0:iyt+1, ix0:ixt+1].copy()
    ny, nx = imc.shape
    # criterion 1
    im0 = imc.copy()
    im0[im0<0.1] = 0.1
    #imm = sf.median(im0, sm.disk(9))
    imm = median_filter(im0, size=6)
    imr = im0/imm
    simg1 = np.zeros_like(imr, dtype=np.int8)
    simg1[np.logical_and(imr>20, im0>int_lim)] = 1
    im0 = imc.copy()
    labelf = label(simg1,connectivity=2, background=0)
    for i in range(np.max(labelf)):
        ind = np.nonzero(labelf == i+1)
        iymax = np.max(ind[0])
        iymin = np.min(ind[0])
        if iymin == 0:
            #print(ind[0], '\n', ind[1])
            #return
            for i in range(ind[0].size):
                im0[ind[0][i], ind[1][i]] = imc[iymax+1, ind[1][i]]
        elif iymax == ny-1:
            for i in range(ind[0].size):
                im0[ind[0][i], ind[1][i]] = imc[iymin-1, ind[1][i]]
        else:
            iymid = (iymax+iymin)*0.5
            for i in range(ind[0].size):
                if ind[0][i] <= iymid:
                    im0[ind[0][i], ind[1][i]] = imc[iymin-1, ind[1][i]]
                else:
                    im0[ind[0][i], ind[1][i]] = imc[iymax+1, ind[1][i]]
    
    # criterion 2
    simg1 = np.zeros_like(im0, dtype=np.int8)
    simg1[im0>=int_lim] = 1
    simg2 = sm.closing(simg1, sm.square(3))
    labelf = label(simg2,connectivity=2, background=0)
    nlabel = np.zeros(np.max(labelf)+1, dtype=np.uint32)
    for i in range(np.max(labelf)+1):
        ind = np.nonzero(labelf == i)
        nlabel[i] = len(ind[0])
    spilab = np.nonzero(nlabel<=n_isolate)
    im1 = im0.copy()
    for i in list(spilab[0]):
        ind = np.nonzero(labelf == i)
        iymax = np.max(ind[0])
        iymin = np.min(ind[0])
        if iymin == 0:
            for i in range(ind[0].size):
                im1[ind[0][i], ind[1][i]] = im0[iymax+1, ind[1][i]]
        elif iymax == ny-1:
            for i in range(ind[0].size):
                im1[ind[0][i], ind[1][i]] = im0[iymin-1, ind[1][i]]
        else:
            iymid = (iymax+iymin)*0.5
            for i in range(ind[0].size):
                if ind[0][i] <= iymid:
                    im1[ind[0][i], ind[1][i]] = im0[iymin-1, ind[1][i]]
                else:
                    im1[ind[0][i], ind[1][i]] = im0[iymax+1, ind[1][i]]
    im2 = im.copy()
    im2[iy0:iyt+1, ix0:ixt+1] = im1
    return im2
