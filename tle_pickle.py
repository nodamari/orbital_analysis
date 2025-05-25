import pickle

if __name__ == '__main__':
    with open("raw_tles.pkl", "rb") as f:
        raw_tle  = pickle.load(f)

    print()