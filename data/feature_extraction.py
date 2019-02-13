from copy import deepcopy


def top_n_term_frequency(ds, fs):
    ds_tf = deepcopy(ds)
    for d in ds_tf:
        for f in fs:
            d['top_n_tf_{f}'.format(f=f.replace(' ', '_'))] = d['tf'].get(f, 0)
    return ds_tf


def top_n_term_occurrence(ds, fs):
    ds_to = deepcopy(ds)
    for d in ds_to:
        for f in fs:
            d['top_n_to_{f}'.format(f=f.replace(' ', '_'))] = bool(d['tf'].get(f, 0))
    return ds_to
