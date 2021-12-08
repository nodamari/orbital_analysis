
import requests
#from bs4 import BeautifulSoup
import datetime as dt
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from tle_obj import TLE

if __name__ == '__main__':
    st = SpaceTrackClient('user', 'pass')
    drange = op.inclusive_range(dt.datetime(2021, 11, 26),  dt.datetime(2021, 11, 27))
    dt_upper = op.less_than(dt.datetime(2021, 11, 27, 12))
    dt_lower = op.greater_than(dt.datetime(2021, 11, 26, 12))
    dt_range = dt_lower + ',' + dt_upper
    dt_list = [dt_lower, dt_upper]
    dt_range = ','.join(dt_list)
    print(dt_range)

    iss_tle = st.tle(norad_cat_id=[25544], epoch=dt_range, limit=1,  format='tle')
    print(iss_tle)
    # clean up tle
    iss_tle = iss_tle.replace('\n','')
    iss_tle = iss_tle.split(' ')
    while '' in iss_tle:
        iss_tle.remove('')

    print(iss_tle)

    iss = TLE(iss_tle)
    iss.true_anomaly()
    iss.semi_major_axis()
    iss.apogee_perigee()
    iss.print()




