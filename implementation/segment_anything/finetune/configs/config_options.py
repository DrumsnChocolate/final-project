import argparse
import copy


class DictAction(argparse.Action):


    @staticmethod
    def _parse_value(val: str):
        if val.startswith('[') or val.startswith('(') or val.startswith('dict('):
            # we just evaluate this string, which is not very safe but very simple. We don't expect any malicious input.
            # if there is malicious input, that's on the user.
            return eval(val)
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

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
                    if i == len(subkeys) - 1:
                        subdict[subkey] = DictAction._parse_value(v)
                        continue
                    if subdict.get(subkey) is None:
                        subdict[subkey] = {}
                    subdict = subdict[subkey]
        setattr(namespace, self.dest, options)
