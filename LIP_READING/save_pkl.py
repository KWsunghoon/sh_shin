import os
import sys
import pickle


class sav_pkl:
    def save_pkl(data, loc, filename):
        loc = os.path.join(loc, filename)
        try:
            with open(loc, 'wb') as file:
                pickle.dump(data, file)
                print("\n\t{0} stored successfully: {1}\n".format(filename, loc))
        
        except Exception as e:
            sys.exit("\terror:{0}, Failed saving the pickle file: {1}\n".format(str(e), loc))