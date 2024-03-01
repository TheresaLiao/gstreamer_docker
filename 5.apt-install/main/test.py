from collections import defaultdict
import numpy as np

# interactiveDangerous_cls = defaultdict(lambda: [])
# interactiveDangerous_cls.update({1: 8, 2: 122, 3: 124,
#                                  4: 125, 7: 126, 11: 128,})
singleDangerous_cls = defaultdict(lambda: [])
singleDangerous_cls.update({0: 7,
                            1: 8,
                            2: 126,
                            3: 125,
                            4: 124,
                            7: 122,
                            11: 128,})


# print(interactiveDangerous_cls.keys())
# print(interactiveDangerous_cls.values())


keys = singleDangerous_cls.keys()
# 
print(list(keys))
print(np.isin(np.array([0,1]),np.array(list(keys))))