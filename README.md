# ALT: Adaptive Law-Based Transformation

**Adaptive Law-Based Transformation (ALT)** is an open-source Python package developed for efficient and accurate time series classification (TSC). ALT leverages adaptive law-based transformations[[1]](#1) to convert raw time series data into a linearly separable feature space, using variable-length shifted time windows. This approach enhances its predecessor, the linear law-based transformation (LLT), by capturing patterns at varying temporal scales with greater precision. ALT achieves state-of-the-art performance in TSC tasks across physics and related domains, all while maintaining minimal computational overhead.

## Repository Structure

- **`ALT.py`**: The core implementation of the Adaptive Law-Based Transformation methodology.
- **`GunPoint_example.ipynb`**: A Jupyter notebook providing a practical example of how to use the ALT package for time series classification. This includes loading data and running the transformation.
- **`results/GunPoint_results.csv`**: Contains the results of running ALT on the [GunPoint](https://www.timeseriesclassification.com/description.php?Dataset=GunPoint) dataset[[2]](#2) as an example.
- **`requirements.txt`**: Lists the required Python packages and dependencies needed to run the code.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Datacompintensive/ALT.git
   cd ALT
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

We recommend familiarizing yourself with the provided example notebook, and modifying that to fit your datasets.

### Loading data 

In the example code we used the aeon module to get the data from the web. You can however load the data from local source. It should be transfromed to a two or three dimensional numpy array or torch tensor. The first should index the instances, the second *(optional)* dimension should index the time series belonging to a given instance *(in case of univariate data this can be omitted)*, and the last one should be the time.

### Initializing ALT

First the parameters have to be set as
```python
R, L, K = 25, 4, 1
extr_methods = [["mean_all"], ["mean", 0.05]]
device = "cuda" if torch.cuda.is_available() else "cpu"
```

To initialize ALT, at least you need to suply the data, and the classes of the instances.
```python
alt = ALT(learn_set, learn_classes, R=R, L=L, K=K, device=device)
```

The class labels should be numbers, preferably integers.
You can choose the device on witch ALT will run, by setting the `device` parameter with a `torch.device` or a string accepted by the `torch.device()` method. 
The `train_length` parameter is for when the data is not uniform length, it should be a list (or equivalent) with the same length as there are insatnces.
Finally you can set the `R`, `L` and `K` parameters, where `R` is the length of the time window for series extraction, `L` is the dimension of embedding, and `K` is the shift between extracted time windows. Each argument can be a single value or a list.  If two or more are given as a list, the length should be the same. The ones supplied with a single value will be padded to a list of suitable length. The corresponding elements of `R` and `L` should satisfy the formula *2l-2|r-1*. Additionally you can set elements of `R` to `None`, then an appropriate l will be computed *(2l-1)*.

### Training the model

The train method trains the model.
```python
alt.train()
```
Its only parameter is `cleanup` which is false by default. If true, after the end of training ALT deletes the date used for training, thus freeing up memory. 

### Saving and loading the model

You can save the trained model with the save method, to load it up after.
*Note: It does not save the data used for training.*

### Transforming data

You can transfrom an instance, with the `transform` method, or a set of instances with the `transform_set` method. For example
```python
transformed_set = alt.transform_set(transform_set, extr_methods=extr_methods,
                                    test_classes=transform_classes, 
                                    save_file_name="results/results.csv", 
                                    save_file_mode="New file")
```

The data to transform should have the shape as described at the training data. `transform` returns with a tensor of features, while `transform_set` returns with a two dimensional tensor of features. If `save_file_name`, and `test_classes` are supplied to the `transform_set` method, it also saves the generated features in csv format. The parameter `save_file_mode` accepts on of the strings from "New file", "Append feature" or "Append instance", and controls the mode of saving.

#### Currently implemented extraction methods

For the `transform` and `transfrom_set` functions you need to specify the list of used extraction methods. The currently implemented methods are:

- `["mean_all"]`: Calculates the average of all the values in a partition.
- `["method", p]`: Calculates the p-th percentile along the rows, then use the method to calculate the final feature.

The following methods are implemented:
- `mean`: average
- `var`: variance
- `excess_kurtosis`: excess kurtosis
- `nth_moment` (the `n` replaced with a positive integer for example "5th_moment")
<br>
After transformation you can analyze the features with Python, MATLAB, or any other program.

## Citation

If you use ALT in your research, please cite the package as follows:

```bibtex
@misc{ALT2025,
  author = {Kurbucz, Marcell T and Haj{\'o}s, Bal{\'a}zs and Halmos, Bal{\'a}zs P and Moln{\'a}r, Vince {\'A} and Jakov{\'a}c, Antal},
  title = {ALT: Adaptive Law-Based Transformation for Time Series Classification},
  year = {2025},
  howpublished = {\url{https://github.com/Datacompintensive/ALT}},
  note = {Python package}
}
```

## Source of data
The data used in the study is sourced from the UCR Time Series Classification Archive[[3]](#3).

## References
<a id="1">[1]</a>
Marcell T. Kurbucz, Balázs Hajós, Balázs P. Halmos, Vince Á. Molnár, Antal Jakovác. (2025).
Adaptive Law-Based Transformation (ALT): A Lightweight Feature Representation for Time Series Classification, [arXiv preprint](https://arxiv.org/abs/2501.09217).

<a id="2">[2]</a>
Ratanamahatana, C. A., & Keogh, E. (2005, April).
Three myths about dynamic time warping data mining. In Proceedings of the 2005 SIAM international conference on data mining (pp. 506-510). Society for Industrial and Applied Mathematics.

<a id="3">[3]</a>
Bagnall, A., Dau, H. A., Lines, J., Flynn, M., Large, J., Bostrom, A., ... & Keogh, E. (2018).
The UEA multivariate time series classification archive, 2018. arXiv preprint arXiv:1811.00075.

## License

This project is licensed under the GPLv3.

## Acknowledgments

We thank the scientific community for providing valuable datasets and benchmarks that guided the development and validation of ALT.
