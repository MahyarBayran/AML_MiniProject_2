from copy import deepcopy


def top_n_frequency(ds, fl, fs):
    ds_tf = deepcopy(ds)
    for d in ds_tf:
        for f in fs:
            d['top_n_fr_{fl}_{f}'.format(fl=fl, f=f.replace(' ', '_'))] = d[fl].get(f, 0)
    return ds_tf


def top_n_occurrence(ds, fl, fs):
    ds_to = deepcopy(ds)
    for d in ds_to:
        for f in fs:
            d['top_n_oc_{fl}_{f}'.format(fl=fl, f=f.replace(' ', '_'))] = bool(d[fl].get(f, 0))
    return ds_to
