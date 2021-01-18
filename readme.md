# AI Trader v1.0


Implementation PPO(Proximal Policy Optimization) algorithm on stock trade.

Based on market data, agent(A.I. trader) is decide to make purchase or sell.
By using policy gradient method, we don't need to build strategies.

I tested it on KOSPI items, but you can use it anywhere you want.
## Getting Started

### Prerequisites 

* pytorch 1.7.0
* pandas
* pandas-datareader
* numpy
* matplotlib

### Installing

```
git clone https://github.com/hojunkim13/aI_trader
```

## Running the train
Info for params #[defalut value]
* sequence_length : Number of days considered in the model #[7]
* amp : amplitude of agent action(order) #[10]


```
.../Train.py
```
## Running the test
Info for params #[defalut value]
* sequence_length : Number of days considered in the model #[7]
* amp : amplitude of agent action(order) #[10]
* period : prediction period from today #[7]
* switch : Test type what you want to run 
    * 'main_test' : Test on setted period & iteration
    * 'find_item' : Test on KOSPI 200 items and save result as csv.
    * 'item_test' : Test on selected stock item.

```
.../Test.py
```



## Contributiong

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.