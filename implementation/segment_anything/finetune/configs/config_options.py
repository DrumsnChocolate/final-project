import argparse
import copy


class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # this does not work all that well with complexer values,
        # but it will have to do for now
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                k, v = kv.split('=')
                subkeys = k.split('.')
                subdict = options
                for i, subkey in enumerate(subkeys):
                    if i == len(subkeys) - 1:
                        subdict[subkey] = v
                        continue
                    subdict[subkey] = {}
                    subdict = subdict[subkey]
        setattr(namespace, self.dest, options)
