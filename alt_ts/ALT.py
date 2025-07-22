import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import os
import gc
import copy
        
    

class ALT:
    """
    Adaptive Law-Based Transformation.

    Finds the preserved quantities of the time series and uses them to
    transform the test instances and prepare them for future classification
    and/or anomaly detection.

    Attributes
    ----------
    train_set : torch.tensor
        The linear laws are based on this time series database.
    train_classes : torch.tensor
        Contains the predefined class labels.
    train_length : torch.tensor
        Length of train instances. Used when the length varies.
    noc : int
        The number of unique classes
    class_labels : torch.tensor
        The list of unique class labelss
    RLK : tuple
        Two dimensional tuple for storing the used r-l-k triplets, where:
        r is the length of the analyzed time window (always a multiple of 2*l-1).
        l is the dimension of the extracted laws.
        k is the step of the time fwindow.
    Ps : dict
        Contains the laws for r-l-k triplets, and a tensor witch indicates witch law belongs to the specific classes.
    tau : int
        Number of training instances.
        (Positive in each case.)
    m : int
        Number of channels.
        (Positive in each case.)
    device : torch.device
        Device where the operations will be carried out. When it is cuda, the results may be needed to put to the cpu for further work. 

    Methods
    -------
    _embed(instance_index, sensor_index, rlk, t=0)
        Creates the time-delay embedding matrix from the time series.
    _get_law(S)
        Returns the linear law corresponding to the embedded matrix.
    _get_P(rlk)
        Stores the laws for an embedding size in tensor `P`.
    _nol(r, k):
        Returns the number of laws for a given r-k pair. 
    train()
        Store tensors into a dictionary for each rlk triplets.
    save(save_file_name):
        Saves the trained model with pickle into the given file. 
    load(load_file_name):
        Static method for loading previously saved models.
    transform(z, extr_methods):
        Transforms one instance (m db time series) into features with the given extraction methods. 
    transform_set(test_set, extr_method, save_file_name, save_file_mode, test_classes):
        Transform a whol set with iterating the transfrom function. If save_file_name is given, saves the collected features in csv format. 
    _save_features(extr_methods, features, test_classes, save_file_name, save_file_mode):
        The inner method for saving features used by transform_set.
    _generate_header(extr_methods):
        The inner method generating the headers for saving.
    _multiply(z, rlk, nol_tilde)
        Embeds, and multiplies an istance with the P matrix corresponding to the given rlk.
    _extract_features(M, extr_methods):
        Extract the features from the result of the multiply function.
    plot(z, rlk, zoom):
        Transforms one instance, and plots the resulting matrix values in ascending order.
    plot_anomalies(z):
        Not implemented yet. Will be used for anomaly detetction
    print_number_of_laws():
        Prints the number of laws.

    Notes
    -----
    One instance contains m number of time series.

    """
    def __init__(self, train_set, train_classes, train_length = None, R=None, L=[5], K=1, device=torch.device('cpu')):
        
        self.device = device
        
        if type(train_set) is np.ndarray:
            train_set = torch.tensor(train_set)
        if type(train_classes) is np.ndarray:
            train_classes = torch.tensor(train_classes)
        if train_set.shape[0] != train_classes.shape[0]:
            raise ValueError("Train classes and train set should have the same length along the first (instance) axis")
        self.train_set = train_set.to(self.device)
        self.train_classes = train_classes.to(self.device)

        if len(self.train_set.shape) == 2:
            self.train_set = torch.unsqueeze(self.train_set, 1)

        if train_length == None:
            self.train_length = torch.full(size=tuple(self.train_classes.shape), fill_value=self.train_set.shape[2], dtype=torch.int)
        elif len(train_length.shape) != 1:
            raise ValueError(f"The train_length expected to have only 1 dimension mot {len(train_length.shape)}")
        elif train_length.shape[0] != train_classes.shape[0]:
            raise ValueError(f"The train_length expected to have the same dimension as train classes({train_classes.shape}). but got {train_length.shape}")
        else:
            self.train_length = train_length
        
        # Number of classes
        self.class_labels = self.train_classes.unique()
        self.noc = self.class_labels.shape[0]

        self.tau, self.m, _ = self.train_set.shape
        
        if type(K) is int and type(L) is int:
            K = [K]
            L = [L]
        elif type(K) is int and type(L) is list:
            L = L
            K = [K]*len(L)
        elif type(K) is list and type(L) is int:
            L = [L]*len(K)
            K = K
        elif type(K) is list and type(L) is list and len(K) == len(L):
            L=L
            K=K
        else:
            raise TypeError(f"Expected two ints, int and a list, or two list with the same size for k and L, but got {type(L)} and {type(K)}")
        
        if R == None:
            R = [L[i]*2-1 for i in range(len(L))]
        elif type(R) is int:
            R = [R]*len(L)
        elif type(R) is list:
            if len(R) == len(L):
                R == R
            else:
                raise ValueError("R and L should have the same length")
        else:
            raise TypeError(f"R should be None, int or list, but got {type(R)}")

        max_R = torch.min(self.train_length).item()

        for i in range(len(L)):
            if type(L[i]) is not int or type(K[i]) is not int:
                raise TypeError("The given `L' and `K' values must be integers.")
            if L[i] < 0 or K[i] < 0 or R[i] < 0:
                raise ValueError("The given `R', `L' and `K' values must be positive.")
            if R[i] > max_R:
                raise ValueError(f"The given r ({R[i]}) and l ({L[i]}) are too large, the maximums are: {max_R} and {(max_R+1)//2}.")
            if (R[i]-1)%(L[i]*2-2) != 0:
                raise ValueError(f"Every R should have the form step*(2*L-2)+1, but got L: {L[i]}, R: {R[i]}")

        self.RLK = tuple((R[i], L[i], K[i]) for i in range(len(L)))

        self.Ps = {}



    def _embed(self, instance_index, sensor_index, rlk, t=0):
        """
        Embeds the given time-window in a real, symmetric matrix

        Parameters
        ----------
        instance_index : int
            The index of the instance to embed
        sensor_index : int
            The index of the time series in the instance to embed
        rlk : tuple
            The (r, l, k) triplet used for the embedding (Note: the k is not important here)
        t : int
            The start index in the time series fr the embedding. The default is 0.
        
        Returns
        -------
        torch.Tensor
            The embedded matrix.
        """
        r, l, _ = rlk
        # Step between datapoints used in the embedding
        step = (r-1)//(2*l-2)
        data = self.train_set[instance_index, sensor_index, t:t+r:step]
        S = torch.stack([data[i:i+l] for i in range(l)])
        return S

    def _get_law(self, S):
        """
        Calculates the eigenvalue-decomposition of the given real, symmetric matrix,
        and returns with the eigenvector corresponding to the smallest absolute valued egienvalue.

        Parameters
        ----------
        S : torch.tensor
            The input matrix.
        
        Returns
        -------
        torch.Tensor
            The eigenvecor.
        
        Raises
        ------
        torch._C._LinAlgError
            If the eigenvalue-decomposition does not converge.
        """
        L, V = torch.linalg.eigh(S)
        L = L**2
        idx = L.argmin()
        return V[:,idx]

    def _get_P(self, rlk):
        """
        Extract the laws for a given (r, l, k) triplet

        Parameters
        ----------
        rlk : tuple
            The given (r, l, k) triplet

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
                - P_classes : torch.Tensor
                    The classes corresponding to each column in P.
                - P : torch.Tensor
                    The generated tensor of laws. 
            
        """
        r, l, k = rlk
        # The nol is an upper bound of the number of laws
        nol = self._nol(r, k)
        # Initializing the P tensor, and P_classes tensor
        P = torch.zeros((l, nol*self.tau, self.m), device=self.device)
        P_classes = torch.full((P.shape[1],), torch.nan, device=self.device)
        for tau_i in range(self.tau):                                             # For each instance
            for m_i in range(self.m):
                for t in range(0, self.train_length[tau_i]-r+1, k):               # For each starting time of embedding
                    S = self._embed(tau_i, m_i, rlk, t)                           # We embed the matrix
                    try:
                        law = self._get_law(S)
                        P[:, tau_i*nol+t//k, m_i] = law
                        P_classes[tau_i*nol+t//k] = self.train_classes[tau_i]     # And store the resulting law
                    except torch._C._LinAlgError as e:
                        # If the eigenvalue-decomposition does not converge
                        print(e)
        # TODO: Remove laws here
        P =  P[:, torch.logical_not(P_classes.isnan()), :]
        P_classes = P_classes[torch.logical_not(P_classes.isnan())]
        return (P_classes, P)

    def _nol(self, r, k):
        """
        Calculates the number of laws from an instance with the given r and k if the instance has the maximal length.

        Parameters
        ----------
        r : int
            The time window
        k : int 
            The step between windows
        
        Returns
        -------
        int
            The number of laws from an instance
        """
        # The maximal length of a time series in the train_set
        t = self.train_set.shape[2]
        return (t - r + 1) // k + 1

    def save(self, save_file_name):
        """
        Saves the trained model

        Parameters
        ----------
        save_file_name : string
            Path of the save file
        """
        model = {"RLK":self.RLK,
                 "m":self.m,
                 "Ps":self.Ps,
                 "noc":self.noc,
                 "class_labels":self.class_labels}
        with open(save_file_name, 'wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(load_file_name):
        """
        Loads the previously terained and saved model.

        Parameters
        ----------
        load_file_name : string
            Path of the save file.

        Returns
        -------
        SBST
            The trained class.
        
        Notes
        -----
            The save does not save the training data, so further training is not possible.
        """
        with open(load_file_name, 'rb') as file:
            model = pickle.load(file)
            sbst = ALT.__new__(ALT)
            sbst.m = model["m"]
            sbst.RLK = model["RLK"]
            sbst.Ps = model["Ps"]
            sbst.noc = model["noc"]
            sbst.class_labels = model["class_labels"]
            return sbst

    def train(self, cleanup = False):
        """
        Trains the model, extract and stores the patterns, hereinafter laws, from the training data,
        for each the given (r, l, k) triplets.

        Parameters
        ----------
        cleanup : bool
            Wheter the training data should be deleted, freeing up memory.
        
        Raises
        ------
        RuntimeError
            If the train was atempted without training data.
        """
        if self.train_set is None:
            raise RuntimeError("The model cannot be trained after loading, as the training data is not saved.")
        for rlk in self.RLK:
            self.Ps[rlk] = self._get_P(rlk)
        if cleanup:
            del self.train_set
            del self.train_classes
            del self.tau
            del self.train_length
            gc.collect()
            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()

                
    def _multiply(self, z, rlk):
        """
        Multiplies an instance with the generated laws.

        Parameters
        ----------
        z : torch.Tensor
            An instance of time series.
        rlk : tuple
            The (r, l, k) triplet.
        
        Returns
        -------
        torch.Tensor
            A tensor of the results.
        """
        r, l, k = rlk
        # Step between datapoints used in the embedding
        step = (r-1)//(2*l-2)
        # The length of the embedding of z along the first dimension
        nol_tilde = (z.shape[1]-step*l+1) // k
        data = torch.stack([z[:, i*k:i*k+step*l:step].T for i in range(nol_tilde)])
        return torch.einsum("kij, ilj -> klj", data, self.Ps[rlk][1])

    def _extract_features(self, M, extr_methods):
        """
        Extract the features from the results of _multiply with all the given extraction methods.

        Parameters
        ----------
        M : torch.Tensor
            The result of the multiplication for one class
        extr_methods : list of lists
            Each element is a two element list, the first being the used statistical method, the second is the used percentile.
            If the second element is not provided, the default 0.05 will be used.

        Returns
        -------
        torch.Tensor
            The calculated features in a two dimensional tensor.

        Raises
        ------
        ValueError
            If the given method is not implemented in ExtractionMethod class.
        """
                    
        results = ExtractMethods.extract(M, extr_methods, self.device)
        return results

    def transform(self, z, extr_methods):
        """
        Transformates, and calculates the features, from an instance, with the given extraction methods.

        Parameters
        ----------
        z : torch.Tensor
            The input time series.
        extr_methods : list of lists
            Each element is a two element list, the first being the used statistical method, the second is the used percentile.
            If the second element is not provided, the default 0.05 will be used.
        
        Returns
        -------
        torch.Tensor
            The one dimensional tensor of the calculated features.

        Raises
        ------
        ValueError
            If the given extraction method is not implemented.
        """
        for i in range(len(extr_methods)):
            if extr_methods[i] in ["mean_all"]:
                extr_methods[i] = extr_methods[i] + [None]
            elif len(extr_methods[i]) == 1:
                extr_methods[i] = extr_methods[i] + [0.05]
        if type(z) is np.ndarray:
            z = torch.tensor(z)
        # Adding second dimension if it is single variate
        if len(z.shape) == 1:
            z = torch.unsqueeze(z, 0)
        if z.shape[0] != self.m:
            raise ValueError(f"The given set has not the same length along the first dimension (got {z.shape[0]}) as the training data ({self.m}).")
        # The number of extraction methods
        n = len(extr_methods)
        # Allocating tensor for features
        feature = torch.zeros((len(self.RLK), self.noc, n, self.m), device=self.device)
        # Extracting features for all (r, l, k)
        for i in range(len(self.RLK)):
            rlk = self.RLK[i]
            M = self._multiply(z, rlk)
            P_classes = self.Ps[rlk][0]
            # Separating the classes
            partitions = [M[:,P_classes==cls,:] for cls in self.class_labels]
            for j in range(self.noc):
                feature[i, j, :, :] = self._extract_features(partitions[j], extr_methods)
        # Flattening the result
        return feature.flatten()

    def transform_set(self, test_set, extr_methods, test_length = None,
                        save_file_name = None, save_file_mode = None, test_classes=None):
        """
        Transforming a whole set of instances, by iterating the transform method.
        If the parameters are given, saves the features in csv format.
        
        Parameters
        ----------
        test_set : torch.Tensor
            The set of instances to transform (similar to the train_set in terms of dimensions)
        extr_methods : list of lists
            Each element is a two element list, the first being the used statistical method, the second is the used percentile.
            If the second element is not provided, the default 0.05 will be used.
        test_length : torch.Tensor
            The useful length of each instance in the set which will be transformed.
        save_file_name : string
            The path for the save file.
            The default is None (not saving).
        save_file_mode : string
            Affects the saving, only used if saving, else None.
            Can be "New_file", "Append feature", or "Append intance", the default is None.
            If the sva_file_name file doesnt exist, this partameter will be ignored.
        test_classes : torch.Tensor
            The classes for the transformed set. Only used if saving, else None.
        
        Returns
        -------
        torch.Tensor
            The results in a two dimensional tensor.
        
        Raises
        ------
        ValueError
            If the given extraction method is not implemented.
        """
        for i in range(len(extr_methods)):
            if extr_methods[i] in [["mean_all"]]:
                extr_methods[i] = extr_methods[i] + [None]
            elif len(extr_methods[i]) == 1:
                extr_methods[i] = extr_methods[i] + [0.05]

        if test_length is None:
            test_length = torch.full((test_set.shape[0], ), self.train_set.shape[2])
        if type(test_set) is np.ndarray:
            test_set = torch.tensor(test_set, dtype=torch.float32)
        if type(test_classes) is np.ndarray:
            test_classes = torch.tensor(test_classes)
        # Calculating features
        if len(test_set.shape) == 2:
            test_set = torch.unsqueeze(test_set, 1)
        if test_set.shape[1] != self.m:
            raise ValueError(f"The given set has not the same length along the second dimension (got {test_set.shape[1]}) as the training data ({self.m}).")
        test_set = test_set.to(self.device)
        # Iterating transform
        features = torch.stack([self.transform(test_set[i, :, :test_length[i]], extr_methods=extr_methods) for i in range(test_set.shape[0])])
        # Saving
        if save_file_name:
            if test_classes != None:
                self._save_features(extr_methods, features, test_classes, save_file_name, save_file_mode)
            else:
                raise TypeError("When saving features you must provide the classes of test instances.")
        return features        


    def _save_features(self, extr_methods, features, test_classes, save_file_name, save_file_mode):
        """
        Saves the generated features in the given path in csv format.

        Parameters
        ----------
        extr_methods : list of list.
            The used extraction methods.
        features : torch.Tensor
            The features generated by transform_set.
        test_classes : torch.Tensor
            The classes of the transformed instances.
        save_file_name : string
            The path of the save file.
        save_file_mode : string
            Can be "New file", "Append feature", or "Append instance".
            If save_file_name file does not exist, this parameter will be ignored.
        
        Raises
        ------
        ValueError
            If the test_classes length doesnt match with the number of test instances.
        
        Notes
        -----
        Makes, or edits a file.
        """
        if features.shape[0] != test_classes.shape[0]:
            raise ValueError(f"test_classes length ({test_classes.shape[0]}) does not match with number of test instances ({features.shape[0]}).")
        features_np = features.cpu().numpy() # convert to Numpy array
        # New file, or if the file doesn't exist.
        if save_file_mode=="New file" or not os.path.exists(save_file_name):
            df = pd.DataFrame(features_np) # convert to a dataframe
            df["Class"] = test_classes
            #n = len(list(extr_methods.keys()))
            feat_list = self._generate_header(extr_methods)
            df.to_csv(save_file_name, index=False, header=feat_list) # save to file
        # Append instance
        elif save_file_mode == "Append instance":
            df = pd.DataFrame(features_np) # convert to a dataframe
            df["Class"] = test_classes
            df.to_csv(save_file_name,  mode='a') # save to file
        # Apppend feature
        elif save_file_mode == "Append feature":
            old_df = pd.read_csv(save_file_name, header=0)
            old_features = old_df.values[:,:-1]
            new_features = np.concatenate((old_features, features_np), axis=1)
            df = pd.DataFrame(new_features) # convert to a dataframe
            #n = len(list(extr_methods.keys()))
            feat_list = list(old_df)[:-1] + self._generate_header(extr_methods)
            df["Class"] = test_classes
            df.to_csv(save_file_name, index=False, header=feat_list) # save to file
        # Default
        else:
            raise ValueError(f"save_file_mode should be either 'New file', 'Append instance' or 'Append feature' but got {save_file_mode}")

    def _generate_header(self, extr_methods):
        """
        Generates the header for the faeture save file.

        Parameters
        ----------
        extr_methods : list of list
            The used extraction methods.
        
        Returns
        -------
        list
            The list of the column names for the pandas dataframe.
        """
        sep = "|"
        return [f"RLK{self.RLK[l_i]}{sep}C{c_i}{sep}F{method}{'' if method in ['mean_all'] else f'(q{q})'}{sep}m{m_i}"\
                                for l_i in range(len(self.RLK))\
                                for c_i in self.class_labels\
                                for method, q in extr_methods\
                                for m_i in range(self.m)]+\
                    ["Class"]


    def print_number_of_laws(self):
        """
        Print the numbver of laws for each class and (r, l, k) triplet.

        Raises
        ------
        RuntimeError
            If called before training.

        Notes
        -----
        Only usable after training.
        """
        # Print the number of laws for each class and each l-k pair
        if self.RLK[0] not in self.Ps.keys():
            raise RuntimeError("The model should be trained first.")
        for rlk in self.RLK:
            r, l, k = rlk
            print(f"Number of laws for r = {r}, l = {l}, k = {k}:")
            print(self.Ps[rlk][0].unique(return_counts=True))


class ExtractMethods:
    """
    Abstract class used for feature extraction.

    Abstract class wich implements the most common statistical methods to be used for feature extraction.

    Methods
    -------
    extract(percentiles, extr_methods, device)
        Extract the features from F with the given extraction methods.
    nth_moment(F, n)
        Calculates the nth_moment.
    excess_kurtosis(percentiles)
    """
    @staticmethod
    def excess_kurtosis(percentiles):
        """
        Calculate the excess kurtosis of the q-th percentile of the squared values along dimension 1.

        Parameters
        ----------
        F : torch.Tensor
            Input tensor.
        q : float, optional
            Percentile to compute, which must be between 0 and 1 inclusive (default is 0.05).

        Returns
        -------
        torch.Tensor
            The fourth_moment of the computed percentiles.
        """
        mean = torch.mean(percentiles, dim=1)
        deviations = percentiles - mean
        fourth_moment = torch.mean(deviations ** 4, dim=1)
        variance = torch.mean(deviations ** 2, dim=1)
        kurt = fourth_moment / (variance ** 2)
        excess_kurtosis = kurt - 3

        # Check for NaN values
        if torch.isnan(excess_kurtosis).any():
            print("Nan values found in the computed excess kurtosis.")
            #raise ValueError("NaN values found in the computed excess kurtosis.")
            return torch.zeros_like(excess_kurtosis)
        return excess_kurtosis

    @staticmethod
    def nth_moment(percentiles, n=4):
        """
        Calculate the n-th moment of the percentiles along dimension 1.

        Parameters
        ----------
        F : torch.Tensor
            Input tensor.
        n : int
            The order of the moment to compute.

        Returns
        -------
        torch.Tensor
            The n-th moment of the computed percentiles.
        """
        mean = torch.mean(percentiles, dim=1)
        deviations = percentiles - mean
        nth_moment = torch.mean(deviations ** n, dim=1)

        # Check for NaN values
        if torch.isnan(nth_moment).any():
            print(f"NaN values found in the computed moment. (n={n})")
            return torch.zeros_like(nth_moment)
        return nth_moment

    @staticmethod
    def extract(F, extr_methods, device = torch.device("cpu")):
        """
        The feature extracting method.

        Parameters
        ----------
        F : torch.Tensor
            Input tensor.
        extr_methods : list of lists
            The used extraction methods
        device : torch.Device
            The device to calculate on.

        Returns
        -------
        torch.Tensor
            The tensor of the collected features
        
        Raises
        ------
        ValueError
            If the given extraction method is not implemented.

        Notes
        -----
        The return tensor has the shape (n, m), where n is the number of used extraction methods, 
        and m is the size of the input tensor along the third dimension.
        """
        F = F**2
        qs = [method[1] for method in extr_methods if method[1] is not None]
        if qs is not []:
            q = torch.Tensor(qs).unique().to(device)
            percentiles = torch.quantile(F, q, dim=1)
        results = []
        for method, perc in extr_methods:
            if method == "mean":
                results.append(torch.mean(percentiles[q == perc], dim = 1))
            elif method == "var":
                var_values = torch.var(percentiles[q == perc], dim = 1)
                if var_values.isnan().any():
                    var_values = torch.zeros_like(var_values)
                    print("Error: NaN values were found in the result of the var calculation!")
                results.append(var_values)
            elif method == "excess_kurtosis":
                results.append(ExtractMethods.excess_kurtosis(percentiles[q == perc]))
            elif method[-7:] == "_moment":
                results.append(ExtractMethods.nth_moment(percentiles[q == perc], n = int(method[:-9])))
            elif method == "mean_all":
                results.append(torch.mean(F, dim=(0, 1)).unsqueeze(0))
            else:
                raise ValueError(f"The method {method} is not implemented")
        #print(list(r.shape for r in results))
        return torch.cat(results)
