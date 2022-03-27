import pandas as pd
import numpy as np
import datetime

def season(x):
  if ((x>=12)&(x<3)):
    return 0
  elif((x>=3)&(x<6)):
    return 1
  elif((x>=6)&(x<9)):
    return 2
  else:
    return 3

def trait_df1(df1):
    dates=[]    
    for k in range(len(df1)):
        mydate = datetime.date(df1.year.iloc[k],df1.Month_Number.iloc[k], df1.day.iloc[k] )
        dates.append(mydate)
    df1.set_index([pd.Index(dates)],inplace=True)
    return df1

def trait_df(l_days,df1):
  df = pd.DataFrame()
  df['new_cases'] = np.NaN
  df['Date'] = l_days

  df["year"] = df["Date"].dt.year
  df["Month_Number"] = df["Date"].dt.month
  df["Week_Day"] = df["Date"].dt.dayofweek
  df['day'] = df["Date"].dt.day
  df['dayofyear'] = df["Date"].dt.dayofyear
  df['quarter'] = df["Date"].dt.quarter
  df['weekend'] = df['Date'].dt.dayofweek >=5
  df['weekend'] = df['weekend'].replace({True : 1 , False : 0}) 
  df['is_month_start'] = df['Date'].dt.is_month_start.astype('int')
  df['is_month_end'] = df['Date'].dt.is_month_end.astype('int')
  df['is_afternoon'] = (df['Date'].dt.hour > 12).astype('int')
  #
  df_tot = pd.concat([df1,df])
  df_tot['season'] = df_tot['Month_Number'].apply(season)
  return df_tot

def predict_cases(A,B,df1,l_days,model):
    last_df_day = datetime.date(df1.year.iloc[len(df1)-1],df1.Month_Number.iloc[len(df1)-1], df1.day.iloc[len(df1)-1] )
    
    if ((A <= last_df_day)&(B > last_df_day)):
        df_tot=trait_df(l_days,df1)

        for k in range(pd.date_range(df_tot.index[len(df1)],df_tot.index[len(df_tot)-1])):
            a_date = df_tot.index[k]
            df_tot['lag_1'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(1)]
            df_tot['lag_2'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(2)]
            df_tot['lag_3'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(3)]
            df_tot['lag_4'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(4)]
            df_tot['lag_5'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(5)]
            df_tot['lag_6'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(6)]
            df_tot['lag_7'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(7)]

        ro = df_tot.loc[a_date]
        ro.drop('new_cases',1,inplace=True)
        pr = model.predict(ro)
        df_tot['new_cases'].iloc[k] = pr



    elif ((A <= last_df_day)&(B <= last_df_day)):
        pred1 = list(df1.loc[A:last_df_day].new_cases.values)

    elif ((A > last_df_day)&(B > last_df_day)):
        l_days = pd.date_range(last_df_day+datetime.timedelta(1) , B)
        df_tot=trait_df(l_days,df1)

        for k in range(pd.date_range(df_tot.index[len(df1)],df_tot.index[len(df_tot)-1])):
            a_date = df_tot.index[k]
            df_tot['lag_1'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(1)]
            df_tot['lag_2'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(2)]
            df_tot['lag_3'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(3)]
            df_tot['lag_4'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(4)]
            df_tot['lag_5'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(5)]
            df_tot['lag_6'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(6)]
            df_tot['lag_7'].iloc[k] = df_tot['new_cases'].iloc[a_date - datetime.timedelta(7)]

        ro = df_tot.loc[a_date]
        ro.drop('new_cases',1,inplace=True)
        pr = model.predict(ro)
        df_tot['new_cases'].iloc[k] = pr
    
    return list(df_tot.loc[A:B].new_cases.values)
    
