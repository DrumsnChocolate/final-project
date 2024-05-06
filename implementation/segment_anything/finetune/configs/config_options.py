import argparse
import copy

import yaml


class DictAction(argparse.Action):


    @staticmethod
    def _parse_value(val: str):
        return yaml.safe_load(val)

    def __call__(self, parser, namespace, values, option_string=None):
        # this does not work all that well with complexer values,
        # but it will have to do for now
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                k, v = kv.split('=', maxsplit=1)
                subkeys = k.split('.')
                subdict = options
                for i, subkey in enumerate(subkeys):
                    if isinstance(subdict, list):  # if tuples add trouble, turn this condition into an or-clause
                        subkey = int(subkey)
                    if i == len(subkeys) - 1:
                        subdict[subkey] = DictAction._parse_value(v)
                        continue
                    try:
                        subdict[subkey]  # raises an error if the key does not exist on the list or dict
                    except KeyError or IndexError:
                        subdict[subkey] = {}  # only passes on dicts or if it is the next index in the list
                    subdict = subdict[subkey]
        setattr(namespace, self.dest, options)
