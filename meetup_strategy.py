#!/usr/bin/env python
# -*- coding: utf-8 -*-

# entropy_functions.py
# (c) Zexun Chen, 2020-03-20
# sxtpy2010@gmail.com

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from datetime import timedelta
from preprocessing import geo2id, pre_processing
import util
from itertools import combinations

SEED = 2020  # set random seed for our random function


class Meetup(object):
    """
    Create a Meetup class to extract useful information from raw csv dataset
    """

    def __init__(self, path, mins_records=200, geoid=False, resolution=None, epsilon=2):
        """
        :arg path: path of source file
        :arg mins_records: the required min number of records for each user
        :arg geoid: whether use geo-coordinates id than placeid
        :arg resolution: if geoid is true, what resolution will be used
        :arg path: string, the location of csv dataset
        :arg epsilon, the shortest length required for predictability computation
        """
        # rdata means raw dataset and pdata means processed dataset
        # since we only needs userid, placieid and datetime in our computation,
        # so these attributes are required.
        self.rdata = pd.read_csv(path)
        self.pdata = pre_processing(self.rdata, min_records=mins_records,
                                    geoid=geoid, resolution=resolution)
        # all the following computations are based on processed data
        self.userlist = sorted(list(set(self.pdata['userid'].tolist())))
        self.user_meetup = None
        self.total_meetup = None
        self.placeidT = None
        self.egolist = None
        self.alterlist = None
        self.epsilon = epsilon

    def _extract_datetime(self, user, latest=False):
        """
        Protect method: extract datetime information for each user
        :param user: string, user we choose
        :return:list, ending time
        """
        if self.placeidT is None:
            self.temporal_placeid()
            self._extract_datetime(user, latest)
        else:
            user_temporal_placeid = self.placeidT[user]
            user_time = pd.to_datetime(user_temporal_placeid.index).to_pydatetime().tolist()
            if latest:
                return user_time[-1]
            else:
                return user_time

    def _former_count(self, ego_end, alter):
        """Return the number of timestamps from alter before the ego_end time"""
        time_alters = self._extract_datetime(alter)
        return sum([x <= ego_end for x in time_alters])

    def find_meetup(self, ego):
        """ Find all the meetups for ego
        :param ego: string, ego's userid
        :return: dataframe, filled with meetup information
        """
        df_ego = self.pdata[self.pdata['userid'] == ego][['userid', 'placeid', 'datetimeH']]
        ego_end = self._extract_datetime(ego, latest=True)

        df_alters = self.pdata[self.pdata['userid'] != ego][['userid', 'placeid', 'datetimeH']]

        """ Here meetup means two users appear in the same placeid at the same time, so we merge two 
        dataframes, keep on placeid and datatime, if they meet, it will be complete row record, 
        otherwise, the record should have NaN. Therefore, we remove all the records with NaN and we
        can have all the meetup information.
        """
        meetup = df_ego.merge(df_alters, how='left', on=['placeid', 'datetimeH']) \
            .dropna()[['userid_x', 'placeid', 'datetimeH', 'userid_y']] \
            .drop_duplicates().groupby(['userid_x', 'userid_y']).size() \
            .reset_index(name='count').sort_values(by=['count', 'userid_y'], ascending=[False, True])

        # compute the percentage
        meetup[['percent']] = meetup[['count']] / meetup['count'].sum()

        alterlist = meetup['userid_y'].tolist()
        alters_former = [self._former_count(ego_end, alter) for alter in alterlist]
        meetup['N_previous'] = np.array(alters_former)
        meetup['N_alter'] = np.array([len(self.placeidT[alter]) for alter in alterlist])

        return meetup

    def all_meetup(self):
        """ concat the meetups for the users
        :return: merged dataframe with all the meetup information
        """
        meetup_list = [self.find_meetup(user) for user in self.userlist]
        user_meetup = pd.concat(meetup_list, sort=False)
        user_meetup = user_meetup.rename(columns={'count': 'meetup'})
        n_user_meetup = user_meetup.groupby('userid_x').size().reset_index(name='count')
        n_user_meetup.columns = ['userid_y', 'n_alter_meetupers']
        self.total_meetup = user_meetup.merge(n_user_meetup, how='left', on='userid_y')

        return self.total_meetup

    def meetup_filter(self, n_meetupers=100, n_previous=200):
        """
        Standing on the total_meetup, only choose some egos who have n_meetupers
        :param n_previous: bool or int, the number of previous checkin is required for alter
        :param n_meetupers: bool or int, the number of meetupers the ego has
        :return: filtered user_meetup
        """
        if self.total_meetup is None:
            self.all_meetup()
            return self.meetup_filter(n_meetupers=n_meetupers, n_previous=n_previous)
        else:
            if n_previous is not None:
                used_meetup = self.total_meetup[self.total_meetup['N_previous'] >= n_previous]
            else:
                used_meetup = self.total_meetup

            meetupers_count = used_meetup.groupby('userid_x')['userid_y'].count().reset_index(name='count')
            if n_meetupers is not None:
                self.egolist = sorted(meetupers_count[meetupers_count['count'] == n_meetupers]['userid_x'].tolist())
            else:
                self.egolist = sorted(meetupers_count['userid_x'].tolist())
            self.user_meetup = used_meetup[used_meetup['userid_x'].isin(self.egolist)]
            self.alterlist = sorted(list(set(self.user_meetup['userid_y'].tolist())))
            self.userlist = sorted(list(set(self.egolist + self.alterlist)))
            self.pdata = self.pdata[self.pdata['userid'].isin(self.userlist)]
            return self.user_meetup

    def temporal_placeid(self, userlist=None):
        """ Extract the time-ordered placeid sequence
        :return: a dictionary, indexed by userid
        """
        if userlist is not None:
            self.userlist = userlist

        self.placeidT = {user: self.pdata[self.pdata['userid'] == user
                                          ].set_index('datetime').sort_index()[['placeid']]
                         for user in self.userlist}
        return self.placeidT

    def _extract_info(self, user):
        """ Protect method: extract temporal-spatial information for each user
        Arg:
            user: string, a userid

        Return:
            user_time: datetime, user's timestamps
            N_uniq_placeid: int, the number user's unique visited placeids
            N_placeid: int, the number of user's visited placeids
            user_placeid: list, time-ordered visited placeid in a list
        """
        if self.placeidT is None:
            self.temporal_placeid()

        user_temporal_placeid = self.placeidT[user]
        user_time = pd.to_datetime(user_temporal_placeid.index).tolist()
        user_placeid = user_temporal_placeid['placeid'].tolist()
        N_uniq_placeid = len(set(user_placeid))
        N_placeid = len(user_placeid)

        return user_time, N_uniq_placeid, N_placeid, user_placeid

    @staticmethod
    def cumulative_cross_entropy(length_ego, alters_L, ave_length):
        """ public method: Compute cumulative cross entropy for an ego and its alters
        Args:
            length_ego: int, the length of the visited placedid sequence of ego.
            alters_L: list, cross-parsed match legnths for the all alters of ego
            ave_length: float, the weighted average lengths of all the users in B

        Return:
            float, cross entropy for a pair of ego and alters
        """
        # remove all the alters_L with nan
        if len(alters_L) == 1:
            sum_L = np.nansum(alters_L)
        else:
            alters_L = list(filter(None, alters_L))
            clean_alters_L = [x for x in alters_L if ~np.isnan(x).all()]
            alters_Lmax = np.amax(clean_alters_L, axis=0)
            sum_L = sum(alters_Lmax)
        return (1.0 * length_ego / sum_L) * np.log2(ave_length)

    @staticmethod
    def weight(ego_L, alter_L=None):
        """ Public method, compute how important of alter for ego
        Args:
            ego_L: list, match length of ego
            alter_L: list, match length of alter compared with ego

        Return: int
            weight of this alter among all the alters compared with ego
        """
        if alter_L is None:
            alter_L = ego_L

        if np.isnan(ego_L).all() | np.isnan(alter_L).all():
            return np.nan
        else:
            # TODO: How to define weight is a problem
            # Definition 1, which is cloe to the paper's definition
            # return sum(x in alter_L for x in ego_L)

            # # Definition 2, count how many times of match length of A find in B
            # count how many elements of ego_L is in alter_L
            return sum(map(lambda x: x > 1, alter_L))

    @staticmethod
    def _ave(lenB, wB):
        """ Compute the average legnth of B
        Args:
            lenB: list, a list of the length of placeid of B, it might be nan
            wB: list, a list of the weight of B, it might be nan
        """
        # since wB is compute by LZ algorithm, it maybe nan
        # lenB is the length of series
        if np.isnan(list(filter(None, wB))).all() | np.isnan(list(filter(None, lenB))).all():
            # another condition is that the product is nan
            # but according to our weight definition,
            # if lenB is nan, wB must be nan
            return np.nan
        else:
            alter_length = np.array(lenB, dtype=np.float64)
            alter_wb = np.array(wB, dtype=np.float64)
            if np.nansum(alter_wb) == 0:
                return np.nan
            else:
                return np.nansum(alter_length * alter_wb) / np.nansum(alter_wb)

    def entropy_predictability(self, length_ego_uni, length_ego, alters_L, ave_length):
        """ Compute entropy and predictability using the definition"""
        CCE_alters = self.cumulative_cross_entropy(length_ego, alters_L, ave_length)
        Pi_alters = util.getPredictability(length_ego_uni, CCE_alters, e=self.epsilon)
        return CCE_alters, Pi_alters

    def _ego_alter_basic(self, ego_time, ego_placeid, ego_L, alter):
        """
        use ego's info and alter id, to compute the basic info for cumulative cross entropy
        :param ego_time: time series of ego
        :param ego_placeid: placeid series of ego
        :param ego_L: match length of ego
        :param alter: userid of alter
        :return:
            L: cross-parsed match length of ego given alter
            wb: the number of times that matches from ego are found in alter
            N_previous: the length of alter happens no later than the last checkin in ego
        """
        alter_time, _, _, alter_placeid = self._extract_info(alter)

        total_time = sorted(ego_time + alter_time)
        PTs = [(total_time.index(x) - ego_time.index(x)) for x in ego_time]

        L = util.LZ_cross_entropy(alter_placeid, ego_placeid, PTs,
                                  lambdas=True, e=self.epsilon)
        wb = self.weight(ego_L, L)
        # only count the length no later than the last time of ego
        length_alter_former = self._length_former(ego_time, alter_time)
        return L, wb, length_alter_former

    @staticmethod
    def _length_former(ego_time, alter_time):
        """
        count how many points of alter_time happen no later than last ego time
        :param ego_time: list of pd.timestamp
        :param alter_time: list of pd.timestamp
        :return: the number of points of alter_time happen no later than last ego time
        """
        ego_end = pd.to_datetime(ego_time).to_pydatetime().tolist()[-1]
        time_alters = pd.to_datetime(alter_time).to_pydatetime().tolist()
        return sum([x <= ego_end for x in time_alters])


class MeetupOneByOne(Meetup):
    """
    Create a Meetup Strategy class based on Meetup class to include all the computation
    """

    def __init__(self, path, mins_records=200, geoid=False, resolution=None, epsilon=2,
                 n_meetupers=100, n_previous=200,
                 user_meetup=None, total_meetup=None, placeidT=None):
        """ MeetupOneByOne needs to have several important inputs
        Arg:
            path, mins_records, geoid, resolution are from the mother class Meetup
            n_meetupers: int, the number of meetupers we set
            n_previous: int, the number of checkins is required
            user_meetup: DataFrame, cols = ['userid_x', 'userid_y', 'meetup', 'percentage']
            placeidT: dict, include all the users' temporal placeid, keys are the userids

        Notes: since user_meetup and placeid need some time to compute, so if possible, you'd better to save them in
        in advance and when you initialise MeetupOneByOne, you can import them as inputs, it will reduce time.
        """
        super(MeetupOneByOne, self).__init__(path, mins_records, geoid, resolution, epsilon)
        if total_meetup is not None:
            self.total_meetup = total_meetup

        self.n_meetupers = n_meetupers

        if user_meetup is None:
            self.user_meetup = self.meetup_filter(n_meetupers=n_meetupers, n_previous=n_previous)
        else:
            self.user_meetup = user_meetup
            # if user_meetup is given directly rather than generating automatically, we have to update egolist,
            # alterlist, userlist and pdata.
            self.egolist = sorted(list(set(self.user_meetup['userid_x'].tolist())))
            self.alterlist = sorted(list(set(self.user_meetup['userid_y'].tolist())))
            self.userlist = sorted(list(set(self.egolist + self.alterlist)))
            self.pdata = self.pdata[self.pdata['userid'].isin(self.userlist)]

        if placeidT is None:
            self.placeidT = self.temporal_placeid()
        else:
            self.placeidT = placeidT

        self.epsilon = epsilon
        self.user_stats = None
        self.ego_stats = None
        self.tr_user_stats = None
        self.tr_ego_stats = None
        self.sr_user_stats = None
        self.sr_ego_stats = None
        self.user_stats_all = None
        self.ego_stats_all = None
        self.total_recency = None

    def _ego_alter_lag(self, ego, lag, egoshow=False):
        """
        extract information of ego and compute all the statistics
        :param ego: userid of ego
        :param lag: the number of hours removed from the previous
        :param egoshow: whether print ego
        :return: CCE and Predictability for alters only and ego+alters case
        """
        #
        ego_time, length_ego_uni, length_ego, ego_placeid = self._extract_info(ego)
        ego_time_delay = [v - timedelta(hours=lag) for v in ego_time]

        ego_L = util.LZ_entropy(ego_placeid, e=self.epsilon, lambdas=True)
        alters = self.user_meetup[self.user_meetup['userid_x'] == ego]['userid_y'].tolist()
        """ego only"""
        total_time = sorted(ego_time_delay + ego_time)
        PTs = [(total_time.index(x) - ego_time_delay.index(x)) for x in ego_time_delay]

        CE_ego = util.LZ_cross_entropy(ego_placeid, ego_placeid, PTs, e=self.epsilon)
        Pi_ego = util.getPredictability(length_ego_uni, CE_ego, e=self.epsilon)

        """alters only"""
        alters_L, wb_length, alters_length = map(list, zip(*[self._ego_alter_basic(ego_time,
                                                                                   ego_placeid,
                                                                                   ego_L,
                                                                                   alter)
                                                             for alter in alters]))
        ave_length = self._ave(alters_length, wb_length)
        CCE_alters, Pi_alters = self.entropy_predictability(length_ego_uni, length_ego,
                                                            alters_L, ave_length)
        """alters + ego"""
        alters_L.append(ego_L)
        alters_length.append(length_ego)
        ego_alters_weight = wb_length + [self.weight(ego_L)]
        ave_length = self._ave(alters_length, ego_alters_weight)
        CCE_ego_alters, Pi_ego_alters = self.entropy_predictability(length_ego_uni,
                                                                    length_ego,
                                                                    alters_L,
                                                                    ave_length)
        if egoshow:
            print(ego)

        return [ego, lag, CE_ego, Pi_ego,
                CCE_alters, CCE_ego_alters,
                Pi_alters, Pi_ego_alters]

    def recency_effect(self, longest_time=24, verbose=False, filesave=False):
        """
        Combine all egos and all time delays as a dataframe
        :param filesave: whether save the results in csv file
        :param longest_time: the longest time delay
        :param verbose: whether display the step
        :return: DataFrame,contains all egos and all time delays
        """
        total_recency = [self._ego_alter_lag(ego, lag, verbose) for lag in range(1, longest_time + 1)
                         for ego in self.egolist]

        self.total_recency = pd.DataFrame(total_recency, columns=['ego', 'delay', 'CE_ego', 'Pi_ego',
                                                                  'CCE_alters', 'CCE_ego_alters',
                                                                  'Pi_alters', 'Pi_ego_alters']
                                          )
        if filesave:
            name = 'recency-effect-' + str(self.n_meetupers) + '-' + str(longest_time) + '.csv'
            self.total_recency.to_csv(name, index=False)

        return self.total_recency

    def _cross_entropy_pair(self, ego_time, ego_placeid, ego_L, alter, alters,
                            L, wb, length_alters, temp_shuffle=False):
        """ Protected method (recursive structure): compute cross entropy related to statistics
        Args:
            ego_time: datetime,
            ego_placeid: list,
            ego_L: list, match length for ego
            alter: string, selected alter
            alters: string list, all the alters for ego
            L: nested list, match legnths for all alters before the selected alter
            wb: list, weights for for all alters before the selected alter
            length_alters: list, length of visited placeids for all alters before the selected alter
            temp_shuffle: bool, whether do shuffle for alter's placeid series

        Return:
            alter related information
        """
        length_ego = len(ego_placeid)
        # length of unique placeid
        length_ego_uni = len(set(ego_placeid))
        alterid = alters.index(alter)

        # included rank is j+1
        rank = alterid + 1

        alter_time, length_alter_uniq, length_alter, alter_placeid = self._extract_info(alter)
        """ Temporal control: given ego, we can find alters, for each alter, we shuffle the sequence of 
        placeid as random """
        if temp_shuffle:
            random.shuffle(alter_placeid)

        alter_log2 = np.log2(length_alter_uniq)
        """Be careful: W1 in cross_entropy is B in the paper, W2 is cross_entropy is A in the paper """
        # so we need to get the relative time order of ego in alter (abosulte position of ego+alter)
        # for function cross_entropy, we need to have PTs
        # here we use LZ-cross entropy, which requires the length at least self.epsilon
        total_time = sorted(ego_time + alter_time)
        PTs = [(total_time.index(x) - ego_time.index(x)) for x in ego_time]

        """ function cross_entropy can return L, as defintion of cumulative cross entropy, we need to get max """
        # compute cross entropy with only this alter
        # Obtain the basic information to extend L, wb, length_alters
        # obtain the cross-parsed match length for this ego-alter pair
        L[alterid] = util.LZ_cross_entropy(alter_placeid, ego_placeid, PTs,
                                           lambdas=True, e=self.epsilon)
        wb[alterid] = self.weight(ego_L, L[alterid])
        # length of alter placeid
        # #length_alters[alterid] = length_alter
        # use length_former to get valid length of alter
        length_alters[alterid] = self._length_former(ego_time, alter_time)

        """ For alter"""
        if wb[alterid] == 0:
            CE_alter, Pi_alter = np.nan, np.nan
        else:
            CE_alter, Pi_alter = self.entropy_predictability(length_ego_uni, length_ego,
                                                             [L[alterid]], length_alters[alterid])

        """ For all above alters """
        # for alters: top above all alters
        alters_L = L[:alterid + 1]
        alters_length = length_alters[:alterid + 1]
        wb_length = wb[:alterid + 1]
        # average lengths
        ave_length = self._ave(alters_length, wb_length)
        # CCE for all above alters
        CCE_alters, Pi_alters = self.entropy_predictability(length_ego_uni, length_ego,
                                                            alters_L, ave_length)
        """For only this alter + ego"""
        # for only this alter and ego
        ego_alter_L = [ego_L, L[alterid]]
        bi_length = np.array([length_alters[alterid], length_ego], dtype=np.float64)
        bi_weight = np.array([wb[alterid], self.weight(ego_L)], dtype=np.float64)
        ave_length = self._ave(bi_length, bi_weight)
        CCE_ego_alter, Pi_ego_alter = self.entropy_predictability(length_ego_uni,
                                                                  length_ego,
                                                                  ego_alter_L,
                                                                  ave_length)
        """For all above alters + ego"""
        # for ego+alters: top above all alters + ego
        alters_L.append(ego_L)
        alters_length.append(length_ego)
        ego_alters_weight = wb[:alterid + 1] + [self.weight(ego_L)]
        ave_length = self._ave(alters_length, ego_alters_weight)
        CCE_ego_alters, Pi_ego_alters = self.entropy_predictability(length_ego_uni,
                                                                    length_ego,
                                                                    alters_L,
                                                                    ave_length)

        """ classify alters as helpful and useless"""
        if CE_alter < np.log2(length_ego_uni):
            group = 'helpful'
        else:
            group = 'useless'

        return [alter, group, rank, wb[alterid], alter_log2,
                CE_alter, CCE_alters, CCE_ego_alter, CCE_ego_alters,
                Pi_alter, Pi_alters, Pi_ego_alter, Pi_ego_alters,
                ]

    def _ego_meetup(self, ego, tempsave=False, egoshow=False,
                    temp_shuffle=False, social_shuffle=False):
        """ Protected method: obtain all the meetup-cross-entropy info for ego
        It can save each ego's record temporarily save to csv file
        Args:
            ego: string, a user
            tempsave: bool, whether it will save csv
            egoshow: bool, whether print ego
            temp_shuffle: bool, whether do temporal control
            social_shuffle: bool, whether do social control
        Return:
            ego with his all alteres' information, DataFrame
        """
        # extraact information of ego and compute all the statistics for all egos
        ego_time, length_ego_uni, length_ego, ego_placeid = self._extract_info(ego)

        # compute the cumulative cross entropy for an ego
        alters = self.user_meetup[self.user_meetup['userid_x'] == ego]['userid_y'].tolist()
        N_alters = len(alters)

        """ Social control: given an ego, we can find its alters. Keep the number of alters, we randomly
        produce alters for ego"""
        if social_shuffle:
            interim = self.userlist.copy()
            interim.remove(ego)
            alters = random.choices(interim, k=N_alters)

        ego_L = util.LZ_entropy(ego_placeid, e=self.epsilon, lambdas=True)
        # initial space
        L = [None] * N_alters
        wb = [None] * N_alters
        length_alters = [None] * N_alters

        ego_stats = [self._cross_entropy_pair(ego_time, ego_placeid, ego_L, alter, alters,
                                              L, wb, length_alters,
                                              temp_shuffle=temp_shuffle) for alter in alters]
        if temp_shuffle:
            ego_stats = pd.DataFrame(ego_stats, columns=[
                'userid_y', 'group', 'Included Rank_tr', 'Weight_tr', 'alter_info_tr',
                'CE_alter_tr', 'CCE_alters_tr', 'CCE_ego_alter_tr', 'CCE_ego_alters_tr',
                'Pi_alter_tr', 'Pi_alters_tr', 'Pi_ego_alter_tr', 'Pi_ego_alters_tr',
            ])
            meetup_ego = ego_stats
            meetup_ego.insert(0, 'userid_x', ego)
        elif social_shuffle:
            ego_stats = pd.DataFrame(ego_stats, columns=[
                'userid_y', 'group', 'Included Rank_sr', 'Weight_sr', 'alter_info_sr',
                'CE_alter_sr', 'CCE_alters_sr', 'CCE_ego_alter_sr', 'CCE_ego_alters_sr',
                'Pi_alter_sr', 'Pi_alters_sr', 'Pi_ego_alter_sr', 'Pi_ego_alters_sr',
            ])
            meetup_ego = ego_stats
            meetup_ego.insert(0, 'userid_x', ego)
        else:
            ego_stats = pd.DataFrame(ego_stats, columns=[
                'userid_y', 'group', 'Included Rank', 'Weight', 'alter_info',
                'CE_alter', 'CCE_alters', 'CCE_ego_alter', 'CCE_ego_alters',
                'Pi_alter', 'Pi_alters', 'Pi_ego_alter', 'Pi_ego_alters',
            ])

            # combine two parts of meetup information
            df_ego_meetup = self.user_meetup[self.user_meetup['userid_x'] == ego]
            meetup_ego = pd.merge(df_ego_meetup, ego_stats, on='userid_y')
            meetup_ego['n_meetupers'] = N_alters

        if tempsave:
            meetup_ego.to_csv('user-meetup-part.csv', index=False, mode='a', header=False)
        if egoshow:
            print(ego)

        return meetup_ego

    def ego_alter_info(self, start=0, end=None, filesave=False, verbose=False,
                       temp_shuffle=False, social_shuffle=False):
        """ Produce all the ego-alter information
        Args:
            start: int, the user started in the userlist, default is 0,
            end: int, the user ended in the userlist, default is the whole dataset
            filesave: bool, whether save the file to csv file
            verbose: bool, whether print the ego in step
            temp_shuffle: bool, whether do the temp_shuffle
            social_shuffle: bool, whether do the social shuffle

        Return:
            user_stats, and also add to the class self oject, self.user_stats
        """
        if end is None:
            end = len(self.egolist)

        random.seed(SEED)
        meetup_list = [self._ego_meetup(ego, tempsave=filesave, egoshow=verbose,
                                        temp_shuffle=temp_shuffle,
                                        social_shuffle=social_shuffle)
                       for ego in self.egolist[start:end]]
        user_stats = pd.concat(meetup_list, sort=False)

        if temp_shuffle:
            self.tr_user_stats = user_stats
            # save the file
            if filesave:
                name = 'user-meetup-info-tr-' + str(self.n_meetupers) + '.csv'
                user_stats.to_csv(name, index=False)
        elif social_shuffle:
            self.sr_user_stats = user_stats
            # save the file
            if filesave:
                name = 'user-meetup-info-sr-' + str(self.n_meetupers) + '.csv'
                user_stats.to_csv(name, index=False)
        else:
            self.user_stats = user_stats
            # save the file
            if filesave:
                name = 'user-meetup-info-' + str(self.n_meetupers) + '.csv'
                user_stats.to_csv(name, index=False)

        return user_stats

    def ego_info(self, start=0, end=None, filesave=False):
        """ Produce all information only related to ego
        Args:
            start: int, the user started in the userlist, default is 0,
            end: int, the user ended in the userlist, default is the whole dataset
            filesave: bool, whether save the file to csv file

        Return:
            ego_stats, and also add to the class self oject, self.ego_stats
        """
        if end is None:
            end = len(self.egolist)

        ego_time, length_ego_uni, length_ego, ego_placeid = zip(*[self._extract_info(ego)
                                                                  for ego in self.egolist[start:end]])
        N = end - start
        ego_LZ_entropy = [util.LZ_entropy(ego_placeid[i], e=self.epsilon) for i in range(N)]
        Pi_ego = [util.getPredictability(length_ego_uni[i], ego_LZ_entropy[i], e=self.epsilon)
                  for i in range(N)]
        ego_log2 = np.log2(list(length_ego_uni))
        df_ego = pd.DataFrame(data={'userid_x': self.egolist[start:end],
                                    'ego_info': ego_log2,
                                    'LZ_entropy': ego_LZ_entropy,
                                    'Pi': Pi_ego
                                    }
                              )
        if self.user_stats is not None:
            df_alters = pd.concat([self.user_stats[self.user_stats['userid_x'] == ego].
                                  tail(1)[['userid_x',
                                           'n_meetupers',
                                           'CCE_alters',
                                           'CCE_ego_alters',
                                           'Pi_alters',
                                           'Pi_ego_alters'
                                           ]] for ego in self.egolist[start:end]
                                   ]
                                  )
            self.ego_stats = df_ego.merge(df_alters, on='userid_x')
            if filesave:
                name = 'user-ego-info-' + str(self.n_meetupers) + '.csv'
                self.ego_stats.to_csv(name, index=False)

        if self.tr_user_stats is not None:
            df_alters_tr = pd.concat([self.tr_user_stats[self.tr_user_stats['userid_x'] == ego].
                                     tail(1)[['userid_x',
                                              'CCE_alters_tr',
                                              'CCE_ego_alters_tr',
                                              'Pi_alters_tr',
                                              'Pi_ego_alters_tr'
                                              ]] for ego in self.egolist[start:end]
                                      ]
                                     )
            self.tr_ego_stats = df_ego.merge(df_alters_tr, on='userid_x')
            if filesave:
                name = 'user-ego-info-tr-' + str(self.n_meetupers) + '.csv'
                self.tr_ego_stats.to_csv(name, index=False)

        if self.sr_user_stats is not None:
            df_alters_sr = pd.concat([self.sr_user_stats[self.sr_user_stats['userid_x'] == ego].
                                     tail(1)[['userid_x',
                                              'CCE_alters_sr',
                                              'CCE_ego_alters_sr',
                                              'Pi_alters_sr',
                                              'Pi_ego_alters_sr'
                                              ]] for ego in self.egolist[start:end]
                                      ]
                                     )
            self.sr_ego_stats = df_ego.merge(df_alters_sr, on='userid_x')
            if filesave:
                name = 'user-ego-info-sr-' + str(self.n_meetupers) + '.csv'
                self.sr_ego_stats.to_csv(name, index=False)

        return self.ego_stats, self.tr_ego_stats, self.sr_ego_stats

    def merge_stats(self, control=False, filesave=False):
        """merge all user-stats and ego-stats if they exist
        Args:
            filesave: whether save the final merged file
            control: whether have temporal and social control
        Return:
            ego+alters all information
        """
        if not control:
            if all(v is not None for v in [self.user_stats, self.ego_stats]):
                ego_alter_stats = self.user_stats.merge(
                    self.ego_stats.copy().drop(columns=['CCE_alters', 'CCE_ego_alters',
                                                        'Pi_alters', 'Pi_ego_alters']),
                    how='left', on=['userid_x'])
                if filesave:
                    name = 'user-meetup-all-' + str(self.n_meetupers) + '.csv'
                    ego_alter_stats.to_csv(name, index=False)

                return ego_alter_stats

        elif all(v is not None for v in [self.user_stats, self.tr_user_stats, self.sr_user_stats]):
            left = self.user_stats.merge(self.tr_user_stats, on=['userid_x',
                                                                 'userid_y']
                                         ).drop(columns=['userid_y']).reset_index(drop=True)
            right = self.sr_user_stats.drop(columns=['userid_y']).reset_index(drop=True)
            all_merge = left.merge(right, left_index=True, right_index=True).drop(columns=['userid_x_y'])

            self.user_stats_all = all_merge.rename(columns={'userid_x_x': 'userid'})

            if filesave:
                name = 'user-meetup-info-all-' + str(self.n_meetupers) + '.csv'
                self.user_stats_all.to_csv(name, index=False)

            if all(v is not None for v in [self.ego_stats, self.tr_ego_stats, self.sr_ego_stats]):
                self.ego_stats_all = self.ego_stats.merge(self.tr_ego_stats, on=['userid_x',
                                                                                 'ego_info',
                                                                                 'LZ_entropy',
                                                                                 'Pi']).merge(
                    self.sr_ego_stats, on=['userid_x', 'ego_info', 'LZ_entropy', 'Pi'])
                if filesave:
                    name = 'user-ego-info-all-' + str(self.n_meetupers) + '.csv'
                    self.ego_stats_all.to_csv(name, index=False)

    def find_meetup_details(self, ego):
        """ Find all the meetups for ego
        :param ego: string, ego's userid
        :return: dataframe, filled with meetup information
        """
        df_ego = self.pdata[self.pdata['userid'] == ego][['userid', 'placeid', 'datetimeH']]
        alterlist = list(set(self.user_meetup[self.user_meetup['userid_x'] == ego]['userid_y'].tolist()))

        df_alters = self.pdata[self.pdata['userid'].isin(alterlist)][
            ['userid', 'placeid', 'datetimeH']]

        """ Here meetup means two users appear in the same placeid at the same time, so we merge two 
        dataframes, keep on placeid and datatime, if they meet, it will be complete row record, 
        otherwise, the record should have NaN. Therefore, we remove all the records with NaN and we
        can have all the meetup information.
        """
        meetup = df_ego.merge(df_alters, how='left', on=['placeid', 'datetimeH']) \
            .dropna()[['userid_x', 'placeid', 'datetimeH', 'userid_y']] \
            .drop_duplicates()

        return meetup

    @property
    def meetup_details(self):
        return pd.concat([self.find_meetup_details(ego) for ego in self.egolist])

    def hist_entropy(self, l=12, w=6, n_bins=100, mode='talk'):
        """ Histogram plot for entropy and more
        :param l: int, long
        :param w: int, width
        :param n_bins: int, how many bins shown in the plot
        :param mode: string, see from seaborn, available,'talk', 'notebook',
        'paper', 'poster'.
        :return: None
        """
        LZentropy = self.ego_stats['LZ_entropy'].dropna()
        CrossEntropy = self.user_stats['CE_alter'].dropna()
        CrossEntropyEgo = self.user_stats['CCE_ego_alter'].dropna()
        CumCrossEntropy = self.ego_stats['CCE_alters'].dropna()
        CumCrossEntropyEgo = self.ego_stats['CCE_ego_alters'].dropna()

        fig, ax = plt.subplots(figsize=(l, w))
        sns.set_context(mode)
        sns.distplot(LZentropy, label='LZ Entropy', bins=n_bins)
        sns.distplot(CrossEntropy, label='Cross Entropy: alter only', bins=n_bins)
        sns.distplot(CrossEntropyEgo, label='Cumulative Cross Entropy: ego + alter', bins=n_bins)
        sns.distplot(CumCrossEntropy, label='Cumulative Cross Entropy: alters only', bins=n_bins)
        sns.distplot(CumCrossEntropyEgo, label='Cumulative Cross Entropy: ego + alters', bins=n_bins)
        # plt.title('Entropy and Cross entropy')
        ax.set(xlabel='Entropy (bits)', ylabel='Density')
        ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
        plt.show()

    def hist_pred(self, l=12, w=6, n_bins=100, mode='talk'):
        """ Histogram plot for predictability and theirs
        :param l: int, long
        :param w: int, width
        :param n_bins: int, how many bins shown in the plot
        :param mode: string, see from seaborn, available,'talk', 'notebook',
        'paper', 'poster'.
        :return: None
        """
        pred = self.ego_stats['Pi'].dropna()
        pred_alter = self.user_stats['Pi_alter'].dropna()
        pred_alter_ego = self.user_stats['Pi_ego_alter'].dropna()
        pred_alters = self.ego_stats['Pi_alters'].dropna()
        pred_alters_ego = self.ego_stats['Pi_ego_alters'].dropna()

        fig, ax = plt.subplots(figsize=(l, w))
        sns.set_context(mode)
        sns.distplot(pred, label=r'$\Pi$: ego', bins=n_bins)
        sns.distplot(pred_alter, label=r'$\Pi$: alter only', bins=n_bins)
        sns.distplot(pred_alter_ego, label=r'$\Pi$: ego + alter', bins=n_bins)
        sns.distplot(pred_alters, label=r'$\Pi$: alters only', bins=n_bins)
        sns.distplot(pred_alters_ego, label=r'$\Pi$: ego + alters', bins=n_bins)
        # plt.title('Entropy and Cross entropy')
        ax.set(xlabel='Predictability $\Pi$', ylabel='Density')
        ax.legend()
        plt.show()

    def paper_hist(self, l=15, w=6, n_bins=100, mode='talk',
                   figsave=False, format='eps'):
        """ Histogram plot for entropy and predictability given by the James' paper
        :param l: int, long
        :param w: int, width
        :param n_bins: int, how many bins shown in the plot
        :param mode: string, see from seaborn, available,'talk', 'notebook',
        'paper', 'poster'.
        :param figsave: whether the figure will be saved
        :param format: png, eps, pdf, and so on.
        :return: None
        """
        LZentropy = self.ego_stats['LZ_entropy'].dropna()
        pred = self.ego_stats['Pi'].dropna()
        # only include the most frequent alter
        CrossEntropy = self.user_stats[self.user_stats['Included Rank'] == 1]['CE_alter'].dropna()
        pred_alter = self.user_stats[self.user_stats['Included Rank'] == 1]['Pi_alter'].dropna()

        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(l, w))
        sns.set_context(mode)
        sns.distplot(LZentropy, label='Entropy', bins=n_bins, ax=ax1, kde=False)
        sns.distplot(CrossEntropy, label='Cross Entropy', bins=n_bins, ax=ax1, kde=False)
        ax1.set(xlabel='Entropy (bits)', ylabel='Count')
        ax1.legend(loc='best')

        sns.distplot(pred, label='Entropy', bins=n_bins, ax=ax2, kde=False)
        sns.distplot(pred_alter, label='Cross Entropy', bins=n_bins, ax=ax2, kde=False)
        ax2.set(xlabel='Predictability', ylabel='Counts')
        ax2.legend(loc='best')
        plt.tight_layout()
        plt.show()

        if figsave:
            title = 'Hist entropy and Pi.' + format
            fig.savefig(title, bbox_inches='tight')

    def paper_interaction(self, threshold=None, interval=None, l=12, w=8, n_bins=200, mode='talk',
                          figsave=False, format='eps'):
        """ Number of meetupers of alter vs predictability
        :param l: int, long
        :param w: int, width
        :param n_bins: int, how many bins shown in the plot
        :param mode: string, see from seaborn, available,'talk', 'notebook',
        'paper', 'poster'.
        :param threshold: int, the largest number of alters included
        :param interval: int, the interval shown in axis
        :param figsave: whether the figure will be saved
        :param format: png, eps, pdf, and so on.

        :return: None
        """
        N_meetupers = self.user_stats.groupby('userid_x')['userid_y'].count().reset_index(name='count')
        iterim = self.user_stats.merge(N_meetupers, left_on='userid_y', right_on='userid_x', how='left')
        if threshold is None:
            threshold = max(iterim['count'])

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True,
                                       figsize=(l, w))
        sns.set_context(mode)
        sns.pointplot(x="count", y="Pi_alter", data=iterim[iterim['count'] <= threshold],
                      ci=95, join=False, ax=ax1)
        if interval is None:
            interval = round(threshold / 20)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(interval))
        ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
        vals = ax1.get_yticks()
        ax1.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
        ax1.set(ylabel='Predictability', xlabel='')
        ax1.set_rasterized(True)

        sns.distplot(iterim[iterim['count'] <= threshold]['count'], bins=n_bins, kde=False, ax=ax2)
        ax2.set(xlabel='Number of meetupers of alter', ylabel='Count')

        fig.tight_layout()
        plt.show()

        if figsave:
            title = 'Meetupers of Alter vs Pi.' + format
            fig.savefig(title, bbox_inches='tight')

    def num_point_plot(self, name, threshold=None, interval=None, l=15, w=6, mode='talk',
                       control=False, figsave=False, format='eps'):
        """ number of included alters vs entropy or predictability
        :param name: string, currently only accept 'entropy' or 'predictability'
        :param threshold: int, the largest number of alters included
        :param interval: int, the interval shown in axis
        :param l: int, long
        :param w: int, width
        :param mode: string, see from seaborn, available,'talk', 'notebook',
        'paper', 'poster'.
        :param control: whether add temporal control and social control in the plot
        :param figsave: whether the figure will be saved
        :param format: png, eps, pdf, and so on.

        :return: None
        """
        fig, ax = plt.subplots(figsize=(l, w))
        sns.set_context(mode)

        if name is 'entropy':
            if control:
                CCE = pd.melt(self.user_stats_all, id_vars=['Included Rank'],
                              value_vars=['CCE_ego_alters', 'CCE_alters',
                                          'CCE_ego_alters_tr', 'CCE_alters_tr',
                                          'CCE_ego_alters_sr', 'CCE_alters_sr'],
                              var_name='CCE')
                baseline = self.ego_stats_all['LZ_entropy'].mean()
                CCE_legend = ['Ego only', 'Alters + ego', 'Alters only',
                              'Alters + ego (TC)', 'Alters only (TC)',
                              'Alters + ego (SC)', 'Alters only (SC)']
                if threshold is None:
                    threshold = len(set(self.user_stats_all['Included Rank'].tolist()))
            else:
                CCE = pd.melt(self.user_stats, id_vars=['Included Rank'],
                              value_vars=['CCE_ego_alters', 'CCE_alters'],
                              var_name='CCE')
                baseline = self.ego_stats['LZ_entropy'].mean()
                CCE_legend = ['Ego only', 'Alters + ego', 'Alters only']
                if threshold is None:
                    threshold = len(set(self.user_stats['Included Rank'].tolist()))

        elif name is 'predictability':
            if control:
                CCE = pd.melt(self.user_stats_all, id_vars=['Included Rank'],
                              value_vars=['Pi_ego_alters', 'Pi_alters',
                                          'Pi_ego_alters_tr', 'Pi_alters_tr',
                                          'Pi_ego_alters_sr', 'Pi_alters_sr'
                                          ],
                              var_name='CCE')
                baseline = self.ego_stats_all['Pi'].mean()
                CCE_legend = ['Ego only', 'Alters + ego', 'Alters only',
                              'Alters + ego (TC)', 'Alters only (TC)',
                              'Alters + ego (SC)', 'Alters only (SC)']
                if threshold is None:
                    threshold = len(set(self.user_stats_all['Included Rank'].tolist()))

            else:
                CCE = pd.melt(self.user_stats, id_vars=['Included Rank'],
                              value_vars=['Pi_ego_alters', 'Pi_alters'],
                              var_name='CCE')
                baseline = self.ego_stats['Pi'].mean()
                CCE_legend = ['Ego only', 'Alters + ego', 'Alters only']
                if threshold is None:
                    threshold = len(set(self.user_stats['Included Rank'].tolist()))

        else:
            raise ValueError('Only available for entropy and predictability')

        sns.pointplot(x="Included Rank", y="value", hue='CCE',
                      data=CCE[CCE['Included Rank'] <= threshold],
                      ci=95, join=False, ax=ax)
        ax.axhline(y=baseline, color='black', linestyle='--', label='Ego')
        leg_handles = ax.get_legend_handles_labels()[0]

        if interval is None:
            interval = round(threshold / 20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        if name is 'predictability':
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
            ax.legend(leg_handles, CCE_legend)
            ax.set(xlabel='Number of included alters', ylabel='Predictability')
        elif name is 'entropy':
            ax.legend(leg_handles, CCE_legend, loc='upper left', bbox_to_anchor=(1.04, 1))
            ax.set(xlabel='Number of included alters', ylabel='Cumulative Cross Entropy')

        plt.show()

        if figsave:
            title = name + '-point.' + format
            fig.savefig(title, bbox_inches='tight')


class MeetupWhole(Meetup):
    """
    Create a Meetup (statistics) class based on Meetup class to include all the computation
    """

    def __init__(self, path, mins_records=200, geoid=False, resolution=None, epsilon=2,
                 n_previous=200,
                 user_meetup=None, total_meetup=None, placeidT=None):
        """
        This class will compute entropy related stats and predictability to describe the dataset
        :param path: dataset path
        :param mins_records: int, the least records of user are required
        :param geoid: bool, whether we use geo-location
        :param resolution: bool or int, what is the resolution when we use geo-location
        :param epsilon: int, the least length is required to compute predictability
        :param n_previous: int, enough checkins are require no later than the last checkin of ego
        :param user_meetup: DataFrame, meetupers network (after n_previous filter)
        :param total_meetup: DataFrame, the whole meetupers network
        :param placeidT: Dict, indexed by ego id and contains all temporal visitations
        """

        super(MeetupWhole, self).__init__(path, mins_records, geoid, resolution, epsilon)
        self.ego_stats = None
        self.ego_stats_gender = None

        if total_meetup is not None:
            self.total_meetup = total_meetup

        if user_meetup is None:
            if n_previous is None:
                self.user_meetup = self.meetup_filter(n_meetupers=None, n_previous=None)
            else:
                self.user_meetup = self.meetup_filter(n_meetupers=None, n_previous=n_previous)

        else:
            self.user_meetup = user_meetup
            # if user_meetup is given directly rather than generating automatically, we have to update egolist,
            # alterlist, userlist and pdata.
            self.egolist = sorted(list(set(self.user_meetup['userid_x'].tolist())))
            self.alterlist = sorted(list(set(self.user_meetup['userid_y'].tolist())))
            self.userlist = sorted(list(set(self.egolist + self.alterlist)))
            self.pdata = self.pdata[self.pdata['userid'].isin(self.userlist)]

        if placeidT is None:
            self.placeidT = self.temporal_placeid()
        else:
            self.placeidT = placeidT

    def _ego_alter(self, ego, egoshow=False):
        """
        extract information of ego and compute all the statistics
        :param ego: userid of ego
        :param egoshow: whether print ego
        :return: CCE and Predictability for alters only and ego+alters case
        """

        ego_time, length_ego_uni, length_ego, ego_placeid = self._extract_info(ego)
        ego_info = np.log2(length_ego_uni)

        ego_L = util.LZ_entropy(ego_placeid, e=self.epsilon, lambdas=True)
        alters = self.user_meetup[self.user_meetup['userid_x'] == ego]['userid_y'].tolist()
        median_meetups = self.user_meetup[self.user_meetup['userid_x'] == ego]['meetup'].median()
        n_meetupers = len(alters)

        """ego only"""
        total_time = sorted(ego_time + ego_time)
        PTs = [(total_time.index(x) - ego_time.index(x)) for x in ego_time]

        CE_ego = util.LZ_cross_entropy(ego_placeid, ego_placeid, PTs, e=self.epsilon)
        Pi_ego = util.getPredictability(length_ego_uni, CE_ego, e=self.epsilon)

        """alters only"""
        alters_L, wb_length, alters_length = map(list, zip(*[self._ego_alter_basic(ego_time,
                                                                                   ego_placeid,
                                                                                   ego_L,
                                                                                   alter)
                                                             for alter in alters]))
        median_valid_len = np.median(alters_length)
        ave_length = self._ave(alters_length, wb_length)
        weighted_ave_len = ave_length
        CCE_alters, Pi_alters = self.entropy_predictability(length_ego_uni, length_ego,
                                                            alters_L, weighted_ave_len)
        """alters + ego"""
        alters_L.append(ego_L)
        alters_length.append(length_ego)
        ego_alters_weight = wb_length + [self.weight(ego_L)]
        ave_length = self._ave(alters_length, ego_alters_weight)
        CCE_ego_alters, Pi_ego_alters = self.entropy_predictability(length_ego_uni,
                                                                    length_ego,
                                                                    alters_L,
                                                                    ave_length)
        if egoshow:
            print(ego)

        return [ego, ego_info,
                n_meetupers, median_meetups, median_valid_len, weighted_ave_len,
                CE_ego, Pi_ego,
                CCE_alters, CCE_ego_alters,
                Pi_alters, Pi_ego_alters]

    def ego_info(self, verbose=False, filesave=False):
        """
        Combine all egos and all time delays as a dataframe
        :param filesave: whether save the results in csv file
        :param verbose: whether display the step
        :return: DataFrame,contains all egos' info
        """

        ego_alters = [self._ego_alter(ego, verbose) for ego in self.egolist]

        self.ego_stats = pd.DataFrame(ego_alters, columns=['ego', 'ego_info',
                                                           'n_meetupers', 'n_meetups(median)',
                                                           'len_alters(median)', 'w_len_alters',
                                                           'CE_ego', 'Pi_ego',
                                                           'CCE_alters', 'CCE_ego_alters',
                                                           'Pi_alters', 'Pi_ego_alters']
                                      )
        if filesave:
            name = 'MeetupWhole.csv'
            self.ego_stats.to_csv(name, index=False)

        return self.ego_stats

    def ego_gender(self, gender_path, filesave=False):
        """
        :param filesave: whether save the file
        :param gender_path: string, the gender dataset path
        :return: DataFrame, contains all egos's info including gender
        """

        if self.ego_stats is None:
            print('Generating ego info now, please wait')
            self.ego_info(verbose=True)

        ego_gender = self._gender_info(gender_path)
        ego_gender_pivot = ego_gender.groupby(['userid_x', 'Gender_Guesser_y']).size().reset_index(
            name='count').pivot(index='userid_x', columns='Gender_Guesser_y', values='count').fillna(0)
        ego_gender_info = pd.DataFrame(ego_gender_pivot.apply(lambda x: x / x.sum(), axis=1).to_records())
        ego_gender_info.columns = ['ego', 'alter:andy', 'alters:femal', 'alter:male',
                                   'alter:mostly_female', 'alter:male', 'alter:unknown']
        self.ego_stats_gender = self.ego_stats.merge(ego_gender_info, how='left', on='ego')
        if filesave:
            name = 'MeetupStatsGender.csv'
            self.ego_stats_gender.to_csv(name, index=False)

        return self.ego_stats_gender

    def _gender_info(self, gender_path):
        name_x = self.user_meetup.apply(lambda row: util.first_name_finder(row.userid_x), axis=1)
        name_y = self.user_meetup.apply(lambda row: util.first_name_finder(row.userid_y), axis=1)
        ego_name = self.user_meetup.assign(First_Name_x=name_x.values, First_Name_y=name_y.values)
        df_gender = pd.read_csv(gender_path)

        ego_gender = ego_name.merge(df_gender, how='left', left_on='First_Name_x', right_on='First_Name').drop(
            columns='First_Name')
        ego_gender = ego_gender.merge(df_gender, how='left', left_on='First_Name_y', right_on='First_Name').drop(
            columns='First_Name').fillna('unknown')
        return ego_gender


class MeetupGender(MeetupWhole):
    """
    Meetup Strategy focusing on Gender aspect
    """

    def __init__(self, path, gender_path, mins_records=200, geoid=False, resolution=None, epsilon=2,
                 n_previous=200,
                 user_meetup=None, total_meetup=None, placeidT=None):
        """
        All inputs are from the father class MeetupWhole and MeetupOneByOne
        """
        super(MeetupGender, self).__init__(path, mins_records, geoid, resolution, epsilon,
                                           n_previous,
                                           user_meetup,
                                           total_meetup,
                                           placeidT)

        meetup_gender = self._gender_info(gender_path)

        # remove all unknown points
        meetup_gender = meetup_gender.replace(['male', 'female', 'mostly_male', 'mostly_female'],
                                              ['M', 'F', 'AM', 'AF'])
        self.user_meetup = meetup_gender[~((meetup_gender['Gender_Guesser_x'].isin(['unknown', 'andy'])) |
                                           (meetup_gender['Gender_Guesser_y'].isin(['unknown', 'andy'])))]
        self.user_stats = {}

    def _ego_meetup(self, ego, meetupers, egoshow=False):

        # extraact information of ego and compute all the statistics for all egos
        ego_time, length_ego_uni, length_ego, ego_placeid = self._extract_info(ego)
        # obtain ego info
        ego_info = np.log2(length_ego_uni)
        ego_LZ_entropy = util.LZ_entropy(ego_placeid, e=self.epsilon)
        Pi_ego = util.getPredictability(length_ego_uni, ego_LZ_entropy, e=self.epsilon)

        # obtain the alters for ego
        alters = meetupers[meetupers['userid_x'] == ego]['userid_y'].tolist()
        N_alters = len(alters)

        ego_L = util.LZ_entropy(ego_placeid, e=self.epsilon, lambdas=True)
        # initial space
        L = [None] * N_alters
        wb = [None] * N_alters
        length_alters = [None] * N_alters

        ego_stats = [self._cross_entropy_pair(ego_time, ego_placeid, ego_L, alter, alters,
                                              L, wb, length_alters) for alter in alters]

        ego_stats = pd.DataFrame(ego_stats, columns=[
            'userid_y', 'group', 'Included Rank', 'Weight', 'alter_info',
            'CE_alter', 'CCE_alters', 'CCE_ego_alter', 'CCE_ego_alters',
            'Pi_alter', 'Pi_alters', 'Pi_ego_alter', 'Pi_ego_alters',
        ])

        # combine two parts of meetup information
        df_ego_meetup = meetupers[meetupers['userid_x'] == ego]
        meetup_ego = pd.merge(df_ego_meetup, ego_stats, on='userid_y')
        meetup_ego['n_meetupers'] = N_alters
        meetup_ego['ego_info'] = ego_info
        meetup_ego['LZ_entropy'] = ego_LZ_entropy
        meetup_ego['Pi'] = Pi_ego

        if egoshow:
            print(ego)
        return meetup_ego

    def ego_alter_info(self, filesave=False, verbose=False, gender=False):
        """ Produce all the ego-alter information
        Args:
            filesave: bool, whether save the file to csv file
            verbose: bool, whether print the ego in step
            gender: string or list of string. Default is False, using all info
        Return:
            user_stats, and also add to the class self oject, self.user_stats
        """
        if not gender:
            meetupers = self.user_meetup
            gender = 'G'
        else:
            if not isinstance(gender, list):
                gender = [gender]
            meetupers = self.user_meetup[self.user_meetup['Gender_Guesser_y'].isin(gender)]

        egolist = sorted(list(set(meetupers['userid_x'].tolist())))
        meetup_list = [self._ego_meetup(ego, meetupers, egoshow=verbose)
                       for ego in egolist]
        user_stats = pd.concat(meetup_list, sort=False)

        name = '_'.join(map(str, gender))
        self.user_stats.update({name: user_stats})

        # save the file
        if filesave:
            name = 'user-meetup-info-' + name + '.csv'
            user_stats.to_csv(name, index=False)

        return user_stats

    def _cross_entropy_pair(self, ego_time, ego_placeid, ego_L, alter, alters,
                            L, wb, length_alters, temp_shuffle=False):
        """ Protected method (recursive structure): compute cross entropy related to statistics
        Args:
            ego_time: datetime,
            ego_placeid: list,
            ego_L: list, match length for ego
            alter: string, selected alter
            alters: string list, all the alters for ego
            L: nested list, match legnths for all alters before the selected alter
            wb: list, weights for for all alters before the selected alter
            length_alters: list, length of visited placeids for all alters before the selected alter
            temp_shuffle: bool, whether do shuffle for alter's placeid series

        Return:
            alter related information
        """
        length_ego = len(ego_placeid)
        # length of unique placeid
        length_ego_uni = len(set(ego_placeid))
        alterid = alters.index(alter)

        # included rank is j+1
        rank = alterid + 1

        alter_time, length_alter_uniq, length_alter, alter_placeid = self._extract_info(alter)
        """ Temporal control: given ego, we can find alters, for each alter, we shuffle the sequence of 
        placeid as random """
        if temp_shuffle:
            random.shuffle(alter_placeid)

        alter_log2 = np.log2(length_alter_uniq)
        """Be careful: W1 in cross_entropy is B in the paper, W2 is cross_entropy is A in the paper """
        # so we need to get the relative time order of ego in alter (abosulte position of ego+alter)
        # for function cross_entropy, we need to have PTs
        # here we use LZ-cross entropy, which requires the length at least self.epsilon
        total_time = sorted(ego_time + alter_time)
        PTs = [(total_time.index(x) - ego_time.index(x)) for x in ego_time]

        """ function cross_entropy can return L, as defintion of cumulative cross entropy, we need to get max """
        # compute cross entropy with only this alter
        # Obtain the basic information to extend L, wb, length_alters
        # obtain the cross-parsed match length for this ego-alter pair
        L[alterid] = util.LZ_cross_entropy(alter_placeid, ego_placeid, PTs,
                                           lambdas=True, e=self.epsilon)
        wb[alterid] = self.weight(ego_L, L[alterid])
        # length of alter placeid
        # #length_alters[alterid] = length_alter
        # use length_former to get valid length of alter
        length_alters[alterid] = self._length_former(ego_time, alter_time)

        """ For alter"""
        if wb[alterid] == 0:
            CE_alter, Pi_alter = np.nan, np.nan
        else:
            CE_alter, Pi_alter = self.entropy_predictability(length_ego_uni, length_ego,
                                                             [L[alterid]], length_alters[alterid])

        """ For all above alters """
        # for alters: top above all alters
        alters_L = L[:alterid + 1]
        alters_length = length_alters[:alterid + 1]
        wb_length = wb[:alterid + 1]
        # average lengths
        ave_length = self._ave(alters_length, wb_length)
        # CCE for all above alters
        CCE_alters, Pi_alters = self.entropy_predictability(length_ego_uni, length_ego,
                                                            alters_L, ave_length)
        """For only this alter + ego"""
        # for only this alter and ego
        ego_alter_L = [ego_L, L[alterid]]
        bi_length = np.array([length_alters[alterid], length_ego], dtype=np.float64)
        bi_weight = np.array([wb[alterid], self.weight(ego_L)], dtype=np.float64)
        ave_length = self._ave(bi_length, bi_weight)
        CCE_ego_alter, Pi_ego_alter = self.entropy_predictability(length_ego_uni,
                                                                  length_ego,
                                                                  ego_alter_L,
                                                                  ave_length)
        """For all above alters + ego"""
        # for ego+alters: top above all alters + ego
        alters_L.append(ego_L)
        alters_length.append(length_ego)
        ego_alters_weight = wb[:alterid + 1] + [self.weight(ego_L)]
        ave_length = self._ave(alters_length, ego_alters_weight)
        CCE_ego_alters, Pi_ego_alters = self.entropy_predictability(length_ego_uni,
                                                                    length_ego,
                                                                    alters_L,
                                                                    ave_length)

        """ classify alters as helpful and useless"""
        if CE_alter < np.log2(length_ego_uni):
            group = 'helpful'
        else:
            group = 'useless'

        return [alter, group, rank, wb[alterid], alter_log2,
                CE_alter, CCE_alters, CCE_ego_alter, CCE_ego_alters,
                Pi_alter, Pi_alters, Pi_ego_alter, Pi_ego_alters,
                ]


class MeetupCrossValid(MeetupWhole):
    """
    Meetup Strategy focusing on adding alters by cross-validation
    """

    def __init__(self, path, mins_records=200, geoid=False, resolution=None, epsilon=2,
                 n_previous=200,
                 user_meetup=None, total_meetup=None, placeidT=None):
        """
        All inputs are from the father class MeetupWhole and MeetupOneByOne
        """
        super(MeetupCrossValid, self).__init__(path, mins_records, geoid, resolution, epsilon,
                                               n_previous,
                                               user_meetup,
                                               total_meetup,
                                               placeidT)
        self.cross_info = None

    def ego_info(self, n_shown=10, verbose=False, filesave=False):
        """
        Combine all egos and all time delays as a dataframe
        :param n_shown: just show n_shown included alters
        :param filesave: whether save the results in csv file
        :param verbose: whether display the step
        :return: DataFrame,contains all egos' info
        """

        ego_alters = pd.concat([self._ego_alter(ego, n_shown, verbose) for ego in self.egolist])
        self.cross_info = ego_alters

        if filesave:
            name = 'MeetupCrossValidation.csv'
            self.cross_info.to_csv(name, index=False)

        return self.cross_info

    def _ego_alter(self, ego, n_shown=10, egoshow=False):
        """
        extract information of ego and compute all the statistics
        :param ego: userid of ego
        :param n_shown: the number of ego is shown
        :param egoshow: whether print ego
        :return: CCE and Predictability for alters only and ego+alters case
        """

        ego_time, length_ego_uni, length_ego, ego_placeid = self._extract_info(ego)

        ego_L = util.LZ_entropy(ego_placeid, e=self.epsilon, lambdas=True)
        total_alters = self.user_meetup[self.user_meetup['userid_x'] == ego]['userid_y'].tolist()
        n_meetupers = len(total_alters)

        """ego only"""
        total_time = sorted(ego_time + ego_time)
        PTs = [(total_time.index(x) - ego_time.index(x)) for x in ego_time]

        CE_ego = util.LZ_cross_entropy(ego_placeid, ego_placeid, PTs, e=self.epsilon)
        Pi_ego = util.getPredictability(length_ego_uni, CE_ego, e=self.epsilon)

        """alters only"""
        """ Here we generate a combinations of included alters"""
        if bool(n_shown) & (n_shown < n_meetupers):
            n_meetupers = n_shown

        CCE_Pi = [self._CCE_Pi(n_included, alters, ego_time, ego_placeid, ego_L,
                               length_ego_uni, length_ego) for n_included in range(1, n_meetupers + 1)
                  for alters in combinations(total_alters, n_included)]

        CCE_Pi = pd.DataFrame(CCE_Pi, columns=['Included',
                                               'CCE_alters', 'CCE_ego_alters',
                                               'Pi_alters', 'Pi_ego_alters']
                              )
        CCE_Pi['Pi_alters_ratio'] = CCE_Pi['Pi_alters'] / Pi_ego
        CCE_Pi['Pi_ego_alters_ratio'] = CCE_Pi['Pi_ego_alters'] / Pi_ego
        CCE_Pi['userid'] = ego

        if egoshow:
            print(ego)

        return CCE_Pi

    def _CCE_Pi(self, n_included, alters, ego_time, ego_placeid, ego_L,
                length_ego_uni, length_ego):
        """ alters only"""
        alters_L, wb_length, alters_length = map(list, zip(*[self._ego_alter_basic(ego_time,
                                                                                   ego_placeid,
                                                                                   ego_L,
                                                                                   alter)
                                                             for alter in alters]))
        ave_length = self._ave(alters_length, wb_length)
        weighted_ave_len = ave_length
        CCE_alters, Pi_alters = self.entropy_predictability(length_ego_uni, length_ego,
                                                            alters_L, weighted_ave_len)
        """alters + ego"""
        alters_L.append(ego_L)
        alters_length.append(length_ego)
        ego_alters_weight = wb_length + [self.weight(ego_L)]
        ave_length = self._ave(alters_length, ego_alters_weight)
        CCE_ego_alters, Pi_ego_alters = self.entropy_predictability(length_ego_uni,
                                                                    length_ego,
                                                                    alters_L,
                                                                    ave_length)
        return [n_included, CCE_alters, CCE_ego_alters, Pi_alters, Pi_ego_alters]
