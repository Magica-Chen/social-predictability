# Meetupers

***Meetupers:*** A basic implementation for validating whether taking advantage of meetup strategy can provide more predictability for our mobility.



The code was tested on Python 3.7. It contains two main classes, **Meetup** and **MeetupStrategy**, and several essential functions.



## Installation

There is no need to install it, just import the function (meetup_strategy.py) at the beginning of your code.



Dependencies: `pandas`, `numpy`, `npmath`, `collections`, `seaborn`, `matplotlib`, `random`.



## Usage

1. Import `meetup_strategy` to your working code at the beginning.

```python
import meetup_strategy as mp
```

2. You should know the path of your raw dataset and make sure you raw dataset has required attributes

**Remark**: since our computations are based on `userid`, `placeid`, `datetime`, please be sure your raw dataset has at least these three attributes.

Of course, if you have similar attributes, but without different names, you **MUST** replace your raw data’s attributes by `userid`, `placeid`, `datetime`. 

For example, if your dataset only has geo-coordinates (longitude and latitude), that’s fine, you can generate `placeid` first using geo-coordinates information and then use this code.

**Updated**: Now you can use geo-coordinates (longitude and latitude) to perform the following experiments. 

3. Obtain meetup information and temporal visited information, for example, [Weeplace dataset](https://drive.google.com/file/d/0BzpKyxX1dqTYYzRmUXRZMWloblU/view),

```python
LetMeet = ms.Meetup(path='data/weeplace_checkins.csv')
# or you can use
# LetMeet = ms.Meetup(path='data/weeplace_checkins.csv', geoid=True)
user_meetup = LetMeet.all_meetup()
user_placeidT = LetMeet.temporal_placeid()
```

4. Define a meetup strategy class

```python
GoMeet = ms.MeetupOneByOne(path='data/weeplace_checkins.csv', user_meetup=user_meetup, placeidT=user_placeidT)
# or using geo-coordinates
# GoMeet = ms.MeetupOneByOne(path='data/weeplace_checkins.csv', geoid=True, user_meetup=user_meetup, placeidT=user_placeidT)
```

In fact, you can define a MeetupOneByOne directly by,

```python
GoMeet = ms.MeetupOneByOne(path='data/weeplace_checkins.csv')
```

The reason why I suggest you define a meetup class first is that finding all the meetup information for this example dataset is very slow (for my personal laptop, it takes approximate 19 hours) . If you import meetup information and temporal information, it will reduce a large amount of time when initialising this MeetupOneByOne.

5. Obtain all the statistics from the view of information theory

```python
GoMeet.ego_alter_info(filesave=True, verbose=True)
```

**Remark**: it also costs a long time, approximate 20 hours for me. So I strongly suggest you set `filesave=True` (save the final result in user-meetup-info.csv) and `verbose=True` (see how you code is running).

The previous code will export a csv file filled with ego-alter pair statistics, if you just need all users’ personal statistics, just use:

```python
GoMeet.ego_info(filesave=True)
```

Of course, if you would like to know more information about user himself/herself, you can easily add some class functions for `ego_info()` function.

6. Think of temporal and social control

The code is to test whether adding a meetuper/meetupers’ information provides more predictability for the user himself/herself. If it is true, at least, (social control) adding a meetuper/meetupers’s information can provide more predictability than adding a random user in the dataset, and (temporal control), adding a meetuper/meetupers’s specific placeid sequence information can provide more predictability than adding a meetuper/meetupers’s shuffled placeid sequence information. 

If social control is required, 

```python
GoMeet.ego_alter_info(filesave=True, verbose=True, social_shuffle=True)
GoMeet.ego_info(filesave=True)
```

It will save `user-meetup-info-sr.csv` and `user-ego-info-sr.csv`.

If temporal control is required,

```python
GoMeet.ego_alter_info(filesave=True, verbose=True, temporal_shuffle=True)
GoMeet.ego_info(filesave=True)
```

It will save `user-meetup-info-tr.csv` and `user-ego-info-tr.csv`.

**Remark**: Since both temporal control and social control need to re-run all the code, you’re not allowed to run social control and temporal control at the same time, that is to say, ***you cannot do***

```python
GoMeet.ego_alter_info(filesave=True, verbose=True, social_shuffle=True, temporal_shuffle=True)
```

7. Merge all the statistics in one document

```python
GoMeet.merge_stats(filesave=True)
```

It will save `user-meetup-info-all.csv` and `user-ego-info-all.csv`

**Remark**: It is only available if `GoMeet` have already had original results, temporal control results, and social control results.

8. Histogram plot for both entropy and predictability

```python
GoMeet.hist_entropy()
GoMeet.hist_pred()
```

9. Plots for comparing entropies and predictabilities with the number of added alters

```python
GoMeet.num_point_plot('entropy', threshold=100, interval=None, l=15, w=6, mode='paper',
                     control=True, figsave=False)
GoMeet.num_point_plot('predictability', threshold=400, interval=None, l=15, w=6, mode='paper', control=True, figsave=False)
```



## More applications

This code can also be used to in many other cases, as long as you modify your raw dataset to have `userid`, `datetime`, `placeid`. 

For example, there are many stocks, AAPL, MSFT, FB, we can have their historical price information (for discrete case, just consider trend, up, draw, down for each day) as time series and we guess if you know MSFT and FB’s historical price time series, we can improve our predictability for AAPL.  To test whether, adding MSFT and FB’s trend series can improve the predictability limit of AAPL, we can just import the raw dataset, change the trend series as `placeid`, set `userid` as the attribute name of the stock, keep the time attribute as `datetime`, and then this code can help you complete. Let’s enjoy it!



Any problem, please let me know (sxtyp2010@gmail.com)!



## Reference



1. Bagrow, James P., Xipei Liu, and Lewis Mitchell. "Information flow reveals prediction limits in online social activity." *Nature human behaviour* 3.2 (2019): 122-128.
2.  Our working paper about social context of human mobility is coming!
