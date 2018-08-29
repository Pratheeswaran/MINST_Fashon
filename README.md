# MINST_Fashon


```
Net(
  (conv1): Conv2d(1, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(5, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (drop_out): Dropout(p=0.2)
  (fc1): Linear(in_features=490, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```
#### Download Dataset from [here](https://www.kaggle.com/zalando-research/fashionmnist)

Test Accuracy of the model on the 10000 test images: 90.36 %
