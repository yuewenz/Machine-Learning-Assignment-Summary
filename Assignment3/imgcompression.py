import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N for black and white images / N * N * 3 for color images
            S: min(N, D) * 1 for black and white images / min(N, D) * 3 for color images
            V: D * D for black and white images / D * D * 3 for color images
        """
        if len(X.shape) == 3:
            X = np.transpose(X, (2,0,1))
        U, S, V = np.linalg.svd(X, full_matrices=True)
        if len(U.shape) == 3:
            U = np.transpose(U, (1,2,0))
            S = S.transpose()
            V = np.transpose(V, (1,2,0))
        return U, S, V


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if len(U.shape) == 2:
            U = np.expand_dims(U, -1)
            S = np.expand_dims(S, -1)
            V = np.expand_dims(V, -1)
        rebuilds = []
        for i in range(U.shape[-1]):
            a = np.matmul(U[:,:k,i], np.diag(S[:k,i]))
            rebuilds.append(np.matmul(a, V[:k,:,i]))
        Xrebuild = np.stack(rebuilds, axis=-1)
        if Xrebuild.shape[-1] == 1:
            Xrebuild = Xrebuild[:,:,0]
        return Xrebuild
        

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in compressed)/(num stored values in original)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        U, S, V = self.svd(X)
        # 1c, n*n, d, d*d
        # 3c, n*n*3, d*3, d*d*3
        n = U.shape[0]
        d = V.shape[0]
        ratio = k * (n + d + 1) / (n * d) 
        return ratio


    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        if len(S.shape) == 1:
            S = np.expand_dims(S, -1)
        S = S ** 2
        recovered_var = np.sum(S[:k, :], axis=0) / np.sum(S, axis=0)
        if len(recovered_var.shape) == 1:
            recovered_var = np.squeeze(recovered_var)
        return recovered_var

        
