import time
import numpy as np

def hopf_map(X):
        """
        Use the Hopf map to project points on the unit sphere in C^2 representing points in CP^1 to points on the unit sphere in R^3.

        Parameters
        ----------
        X : complex ndarray(n,2)
            Points on the unit sphere in C^2.

        Returns
        -------
        real ndarray(n,3)
            Points on the unit sphere in R^3.

        """
        Y = np.zeros((2 * X.shape[1], X.shape[0]))
        Y[::2, :] = np.real(X).T
        Y[1::2, :] = np.imag(X).T
        return np.array(
            [
                2 * (np.prod(Y[[0, 2], :], axis=0) + np.prod(Y[[1, 3], :], axis=0)),
                2 * (np.prod(Y[[1, 2], :], axis=0) - np.prod(Y[[0, 3], :], axis=0)),
                np.sum(Y[[0, 1], :] ** 2, axis=0) - np.sum(Y[[2, 3], :] ** 2, axis=0),
            ]
        ).T
    
def ppca(class_map, proj_dim, projective_dim_red_mode="one-by-one", verbose=False):
        """
        Principal Projective Component Analysis (Jose Perea 2017)

        Parameters
        ----------
        class_map : ndarray (N, d)
            For all N points of the dataset, membership weights to
            d different classes are the coordinates
        proj_dim : integer
            The dimension of the projective space onto which to project
        projective_dim_red_mode : string
            Either "one-by-one", "exponential", or "direct". How to perform equivariant
            dimensionality reduction. "exponential" usually works best, being fast
            without compromising quality.
        verbose : boolean
            Whether to print information during iterations

        Returns
        -------
        {'variance': ndarray(N-1)
            The variance captured by each dimension
        'X': ndarray(N, proj_dim+1)
            The projective coordinates
        }

        """
        if verbose:
            print(
                "Doing ppca on %i points in %i dimensions down to %i dimensions"
                % (class_map.shape[0], class_map.shape[1], proj_dim)
            )

        assert projective_dim_red_mode in ["direct", "exponential", "one-by-one"]

        X = class_map.T
        variance = np.zeros(X.shape[0] - 1)
        n_dim = class_map.shape[1]

        def _one_step_linear_reduction(X, dims_to_keep):
            try:
                _, U = np.linalg.eigh(X.dot(np.conjugate(X).T))
                U = np.fliplr(U)
            except:
                U = np.eye(X.shape[0])
            Y = (np.conjugate(U).T).dot(X)
            Y = Y[:dims_to_keep, :]
            X = Y / np.linalg.norm(Y, axis=0)[None, :]
            return X

        total_dims_to_keep = proj_dim + 1

        mode = projective_dim_red_mode
        if mode == "direct":
            XRet = _one_step_linear_reduction(X, total_dims_to_keep)
        elif mode == "exponential":
            to_keep_this_iter = (n_dim - total_dims_to_keep) // 2
            while to_keep_this_iter > 0:
                X = _one_step_linear_reduction(
                    X, total_dims_to_keep + to_keep_this_iter
                )
                to_keep_this_iter = to_keep_this_iter // 2
            if X.shape[0] > total_dims_to_keep:
                X = _one_step_linear_reduction(X, total_dims_to_keep)
            XRet = X

        elif mode == "one-by-one":
            tic = time.time()
            # Projective dimensionality reduction : Main Loop
            XRet = None
            for i in range(n_dim - 1):
                if i == n_dim - proj_dim - 1:
                    XRet = X
                try:
                    _, U = np.linalg.eigh(X.dot(np.conjugate(X).T))
                    U = np.fliplr(U)
                    # U, _, _ = np.linalg.svd(X)
                except:
                    U = np.eye(X.shape[0])
                variance[-i - 1] = np.mean(
                    (np.pi / 2 - np.real(np.arccos(np.abs(U[:, -1][None, :].dot(X)))))
                    ** 2
                )
                Y = (np.conjugate(U).T).dot(X)
                # y = np.array(Y[-1, :])
                Y = Y[0:-1, :]
                # X = Y / np.sqrt(1 - np.abs(y) ** 2)[None, :]
                X = Y / np.linalg.norm(Y, axis=0)[None, :]
            if verbose:
                print("Elapsed time ppca: %.3g" % (time.time() - tic))

        # Return the variance and the projective coordinates
        return {"variance": variance, "X": XRet.T}