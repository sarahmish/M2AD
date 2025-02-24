# M2AD
Unsupervised Anomaly Detection for Heterogeneous Multivariate Time Series Data from Multiple Systems.

This is a code repository for the papar "M2AD: Detecting Anomalies in Heterogeneous Multivariate Time Series from Multiple Systems" that will appear in AISTATS 2025.

## Quickstart
It is recommended to create a new environment using python 3.10 or 3.11 to install the package.

```
pip install -r requirements.txt
```

### Dataset
We will be using the dataset made public by NASA, please visit [telemanom github](https://github.com/khundman/telemanom) to acquire the data.

### Example Run

```python3
signal = 'S-1'

data = pd.read_csv(f'{signal}-train.csv')
data.head()
````
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1222819200</td>
      <td>-0.366359</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1222840800</td>
      <td>-0.394108</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1222862400</td>
      <td>0.403625</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1222884000</td>
      <td>-0.362759</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1222905600</td>
      <td>-0.370746</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>


Once we have loaded the data, we create an `M2AD` instance
```python3
time_column = 'timestamp'
sensor_columns = ['0']
covariate_columns = list(map(str, range(1, len(data.columns)-1)))

model = M2AD(dataset='SMAP', entity=signal, 
             time_column=time_column, 
             sensors=sensor_columns, 
             covariates=covariate_columns, 
             epochs=30, 
             error_name='area',
             feature_range=(0, 1))
```

In the definition of the class, we pass the timestamp column, the sensor columns, and the covariates columns. Based on this dataset, we only have one sensor column and the remaining are covariates.

To train the model, simply run `fit` function
```python
model.fit(data, tolerance=5)
```

Once the model finished training, we load the testing data and call `detect` to find anomalies.
```python3
test = pd.read_csv(f'{signal}-test.csv')

anomalies = model.detect(test)
````

```bash
>>> anomalies
    dataset  entity       start          end    score
0      SMAP     S-1  1399356000   1404540000    0.045
```

The result will show the timestamps where the model detected the observation to be anomalous.

## Citation
```
@inproceedings{m2ad,
  title={M2AD: Detecting Anomalies in Heterogeneous Multivariate Time Series from Multiple Systems},
  author={Alnegheimish, Sarah and He, Zelin and Reimherr, Matthew and Chandrayan, Akash and Pradhan, Abhinav and D'Angelo, Luca},
  booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2025}
}
```