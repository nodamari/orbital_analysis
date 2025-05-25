from spacetrack import SpaceTrackClient
import spacetrack.operators as op
import datetime
import pickle

if __name__ == '__main__':
    """
    This script is for getting raw TLEs from space-track.org and storing them into a pickle file for later use.
    Excessively querying the API will result in an account suspension.
    """
    # get username and password from userpass.txt file
    with open('userpass.txt') as f:
        contents = f.readlines()
    user = contents[0].rstrip('\n')
    password = contents[1]

    st = SpaceTrackClient(user, password)
    sat_cat_id = 55473#270272#58776 # Starlink-1117

    # set epoch range as desired
    drange = op.inclusive_range(datetime.datetime(2025, 4, 20, 0), datetime.datetime(2025, 5, 24, 15))
    # query TLEs
    raw_tles = st.tle(epoch=drange, format="tle", limit=12, norad_cat_id=sat_cat_id)

    print(raw_tles)
    with open(f"raw_tles_{sat_cat_id}.pkl", "wb") as f: # might be good to use unique names for the pickle file depending on query
        pickle.dump(raw_tles, f)
