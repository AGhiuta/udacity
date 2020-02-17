import cv2
import numpy as np


def disparity_ssd(L, R, tplSize=[11, 11], disparity=30):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    # TODO: Your code here
    rows, cols = L.shape
    tplRows, tplCols = tplSize
    padV, padH = tplRows >> 1, tplCols >> 1
    halfD = disparity >> 1

    # Initialize disparity map
    D = np.zeros(L.shape, dtype=np.float32)

    # Add 0 padding for the left and right images
    padL = np.pad(L, ((padV, padV), (padH, padH)),
                  'constant', constant_values=0).astype(np.float32)
    padR = np.pad(R, ((padV, padV), (padH, padH)),
                  'constant', constant_values=0).astype(np.float32)

    for row in range(rows):
        for col in range(cols):
            # Extract the the template from the left image
            tpl = padL[row:row+tplRows,
                       col:col+tplCols]
            RStripMinCol = max(col-halfD, 0)
            # Extract a search strip of len=disparity around the current column
            RStrip = padR[row:row+tplRows, RStripMinCol:col+halfD+1]

            # Compute template match errors
            err = cv2.matchTemplate(RStrip, tpl, method=cv2.TM_SQDIFF)
            err = cv2.normalize(err, err, alpha=0, beta=1,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # Compute the disparity
            _, _, minLoc, _ = cv2.minMaxLoc(err)
            D[row, col] = minLoc[0] + RStripMinCol - col

    return D
