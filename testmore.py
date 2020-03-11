import feather as ft
import numpy as np
import matplotlib.pyplot as plt
from math import log, e
import time
import pickle
from meetup_process import MeetupStrategy
import pandas as pd

""" Mobility dataset import"""
df_wp = ft.read_dataframe('data/weeplace_checkins_without_loc_NA.feather') # it is the dataset without NA location

# # it will be the same if you use the original csv file.
# df_wp= pd.read_csv('data/weeplace_checkins.csv')  # this is original Weeplace dataset without any processing, including some NA location
# df_wp = df_wp.dropna(subset=['placeid', 'userid', 'datetime'])

"""Previous results of meetup information import"""
pickle_in = open("meetup_store.pickle", "rb")
meetup_store = pickle.load(pickle_in)
pickle_in.close()

user_list = list(set(df_wp['userid'].tolist()))

user_meetup=pd.concat(meetup_store)
user_meetup=user_meetup.rename(columns = {'count':'meetup'})

# read picle file
# pickle_in = open("user_placeidT.pickle", "rb")
# user_placeidT = pickle.load(pickle_in)
# pickle_in.close()

we_meet = MeetupStrategy(user_list, user_meetup, user_placeidT)
we_meet.ego_alter_info(end=2)